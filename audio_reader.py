import os
import random
import re
import threading
import librosa
import numpy as np
import tensorflow as tf


def load_audio_alignments(alignment_list_file, sample_rate):
    '''Load the audio waveforms and alignments from a list file.
       The file format is
       wav_path user_# : phone#_1 ... phone#_N : log_f0_1 .. log_f0_N
       where phone#_t* ints are per-frame phone labels at 100 frames/second
       and log_f0_* are per-frame log-f0 values.
    '''
    assert sample_rate % 100 == 0        # We'll need this.

    epoch = 0
    files = []
    alignments = {}
    iphone = iuser = 0

    with open(alignment_list_file) as f:
        for line in f:
            a = line.rstrip().split()
            path = a.pop(0)
            user = int(a.pop(0))
            if user >= iuser:
                iuser = user+1
            assert a.pop(0) == ':'
            alen = (len(a) - 1)//3
            assert a[alen*2] == ':'
            frame_labels = np.array(map(int, a[0:alen*2]), dtype=np.int32)
            frame_lf0 = np.array(map(float, a[alen*2+1:]), dtype=np.float32)
            for i, phone in enumerate(frame_labels):
                if phone >= iphone:
                    iphone = phone+1
            frame_labels = frame_labels.reshape(-1, 2)
            files.append(path)
            alignments[path] = user, frame_labels, frame_lf0
    print("files length: {} users {} phones {}".format(
        len(files), iuser, iphone))
    return files, alignments, iuser, iphone


# Never finishes.
def audio_iterator(files, alignments, sample_rate, n_mfcc):
    epoch = 0

    while True:
        random.shuffle(files)
        for filename in files:
            user_id, frame_labels, frame_lf0 = alignments[filename]
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
            # normalize audio
            maxv = np.max(np.abs(audio))
            if maxv > 1e-5:
                audio *= 1.0/maxv
            repeat_factor = sample_rate/100
            sample_labels = frame_labels.repeat(repeat_factor, axis=0)
            sample_lf0 = frame_lf0.repeat(repeat_factor)
            audio = audio[:sample_labels.shape[0]]  # clip off the excess.
            user = np.full((sample_labels.shape[0],), user_id, dtype=np.int32)
            mfcc = librosa.feature.mfcc(
                audio[:-1], sr=sample_rate, n_mfcc=n_mfcc,
                hop_length=repeat_factor, n_fft=400).transpose()
            mfcc = mfcc.repeat(repeat_factor, axis=0)
            assert len(audio) == len(sample_labels) == len(user) == \
                mfcc.shape[0]
            yield filename, audio, user, sample_labels, sample_lf0, mfcc

        print "Epoch {} ended".format(epoch)
        epoch += 1


def trim_silence(audio, user, alignment, lf0, mfcc, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array if the whole audio was silence.
    if indices.size:
        audio = audio[indices[0]:indices[-1]]
        user = user[indices[0]:indices[-1]]
        alignment = alignment[indices[0]:indices[-1], :]
        lf0 = lf0[indices[0]:indices[-1]]
        mfcc = mfcc[indices[0]:indices[-1], :]
    else:
        audio = audio[0:0]
        user = user[0:0]
        alignment = alignment[0:0, :]
        lf0 = lf0[0:0]
        mfcc = mfcc[0:0, :]
    return audio, user, alignment, lf0, mfcc


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, alignment_list_file, coord, sample_rate,
                 chunk_size, overlap=0, reverse=False, silence_threshold=None,
                 n_chunks=5, queue_size=5, n_mfcc=12):

        assert chunk_size > overlap

        self.coord = coord
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.reverse = reverse
        self.silence_threshold = silence_threshold
        self.n_chunks = n_chunks
        self.overlap = overlap
        self.n_mfcc = n_mfcc
        self.context = 2        # Hard coded for now.
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.user_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.align_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.lf0_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.mfcc_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(
            queue_size,
            ['float32', 'int32', 'int32', 'float32', 'float32'],
            shapes=[(None,), (None,), (None, self.context), (None,),
                    (None, self.n_mfcc)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.user_placeholder,
                                           self.align_placeholder,
                                           self.lf0_placeholder,
                                           self.mfcc_placeholder])

        self.files, self.alignments, self.n_users, self.n_phones = \
            load_audio_alignments(alignment_list_file, sample_rate)

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    # Thread main is a little tricky.  We want to enqueue multiple chunks,
    # each from a separate utterance (so that we have speaker diversity
    # for each training minibatch.
    # We keep an array of buffers for this.  We cut fixed sized chunks
    # out of the buffers.  As each buffer exhausts, we load a new
    # audio file (using audio_iterator) and concatenate it with the
    # buffer remnants.
    def thread_main(self, sess):
        # buffers: the array of buffers.
        buffers = [(np.array([], dtype=np.float32),
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32).reshape(0, self.context),
                    np.array([], dtype=np.float32),
                    np.array([], dtype=np.float32).reshape(0, self.n_mfcc)
                    )]*self.n_chunks
        # iterator.next() will never stop.  It will allow us to go
        # through the data set multiple times.
        iterator = audio_iterator(self.files, self.alignments,
                                  self.sample_rate, self.n_mfcc)

        # Inflate chunk_size by the amount of overlap for convenience:
        orig_chunk_size = self.chunk_size
        padded_chunk_size = orig_chunk_size + self.overlap

        stop = False
        while not stop:
            # The buffers array has 3 elements per entry:
            # 1) audio.  2) user ID. 3) Phone alignments.
            for i, (buffer_, buf_user, buf_align, buf_lf0, buf_mfcc) in \
                    enumerate(buffers):
                if self.coord.should_stop():
                    stop = True
                    break

                assert len(buffer_) == len(buf_user) == buf_align.shape[0] == \
                    len(buf_lf0) == buf_mfcc.shape[0]

                # Cut samples into fixed size pieces.
                # top up the current buffers[i] element if it
                # is too short.
                while len(buffer_) < padded_chunk_size:
                    filename, audio, user, alignment, lf0, mfcc = \
                        iterator.next()
                    if self.silence_threshold is not None:
                        # Remove silence
                        audio, user, alignment, lf0, mfcc = \
                            trim_silence(audio, user, alignment, lf0, mfcc,
                                         self.silence_threshold)

                        if audio.size == 0:
                            print("Warning: {} was ignored as it contains "
                                  "only silence. Consider decreasing "
                                  "trim_silence threshold, or adjust volume "
                                  "of the audio.".format(filename))

                    if not self.reverse:
                        buffer_ = np.append(buffer_, audio)
                        buf_user = np.append(buf_user, user)
                        buf_align = np.append(buf_align, alignment, axis=0)
                        buf_lf0 = np.append(buf_lf0, lf0)
                        buf_mfcc = np.append(buf_mfcc, mfcc, axis=0)
                    else:
                        buffer_ = np.append(audio, buffer_)
                        buf_user = np.append(user, buf_user)
                        buf_align = np.append(alignment, buf_align, axis=0)
                        buf_lf0 = np.append(lf0, buf_lf0)
                        buf_mfcc = np.append(mfcc, buf_mfcc, axis=0)

                # Send one piece
                if not self.reverse:
                    piece = buffer_[:padded_chunk_size]
                    piece_user = buf_user[:padded_chunk_size]
                    piece_align = buf_align[:padded_chunk_size, :]
                    piece_lf0 = buf_lf0[:padded_chunk_size]
                    piece_mfcc = buf_mfcc[:padded_chunk_size, :]
                    buffer_ = buffer_[orig_chunk_size:]
                    buf_user = buf_user[orig_chunk_size:]
                    buf_align = buf_align[orig_chunk_size:, :]
                    buf_lf0 = buf_lf0[orig_chunk_size:]
                    buf_mfcc = buf_mfcc[orig_chunk_size:, :]
                else:
                    piece = buffer_[-padded_chunk_size:]
                    piece_user = buf_user[-padded_chunk_size:]
                    piece_align = buf_align[-padded_chunk_size:, :]
                    piece_lf0 = buf_lf0[-padded_chunk_size:]
                    piece_mfcc = buf_mfcc[-padded_chunk_size:, :]
                    buffer_ = buffer_[:-orig_chunk_size]
                    buf_user = buf_user[:-orig_chunk_size]
                    buf_align = buf_align[:-orig_chunk_size, :]
                    buf_lf0 = buf_lf0[:-orig_chunk_size]
                    buf_mfcc = buf_mfcc[:-orig_chunk_size, :]
                sess.run(
                    self.enqueue,
                    feed_dict={self.sample_placeholder: piece,
                               self.user_placeholder: piece_user,
                               self.align_placeholder: piece_align,
                               self.lf0_placeholder: piece_lf0,
                               self.mfcc_placeholder: piece_mfcc})
                buffers[i] = (buffer_, buf_user, buf_align, buf_lf0, buf_mfcc)

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
