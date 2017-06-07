import os
import random
import re
import threading
import librosa
import numpy as np
import tensorflow as tf


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def load_audio_alignments(alignment_list_file, sample_rate):
    '''L:oad the audio waveforms and alignments from a list file.
       The file format is
       wav_path user_# : phone#_1 phone#_2 ... phone#_N
       where phone#_t* ints are per-frame phone labels at 100 frames/second.
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
            frame_labels = np.array(map(int, a), dtype=np.int32)
            for i, phone in enumerate(frame_labels):
                if phone >= iphone:
                    iphone = phone+1
            files.append(path)
            alignments[path] = frame_labels, user
    print("files length: {} users {} phones {}".format(
        len(files), iuser, iphone))
    return files, alignments, iuser, iphone


# Never exits.
def audio_iterator(files, alignments, sample_rate):
    epoch = 0

    while True:
        random.shuffle(files)
        for filename in files:
            frame_labels, user_id = alignments[filename]
            audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
            sample_labels = frame_labels.repeat(sample_rate/100)
            audio = audio[:sample_labels.shape[0]]  # clip off the excess.
            user = np.full((sample_labels.shape[0],), user_id, dtype=np.int32)
            assert len(audio) == len(sample_labels) == len(user)
            yield filename, audio, user, sample_labels

        print "Epoch {} ended".format(epoch)
        epoch += 1


def trim_silence(audio, user, alignment, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array if the whole audio was silence.
    if indices.size:
        audio = audio[indices[0]:indices[-1]]
        user = user[indices[0]:indices[-1]]
        alignment = alignment[indices[0]:indices[-1]]
    else:
        audio = audio[0:0]
        user = user[0:0]
        alignment = alignment[0:0]
    return audio, alignment, user


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, alignment_list_file, coord, sample_rate,
                 chunk_size, reverse=False, silence_threshold=None,
                 n_chunks=5, queue_size=5):
        self.coord = coord
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.reverse = reverse
        self.silence_threshold = silence_threshold
        self.n_chunks = n_chunks
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.user_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.align_placeholder = tf.placeholder(dtype=tf.int32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32', 'int32', 'int32'],
                                         shapes=[(None,), (None,), (None,)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder,
                                           self.align_placeholder,
                                           self.user_placeholder])

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
                    np.array([], dtype=np.int32))]*self.n_chunks
        # iterator.next() will never stop.  It will allow us to go
        # through the data set multiple times.
        iterator = audio_iterator(self.files, self.alignments,
                                  self.sample_rate)
        stop = False
        while not stop:
            # The buffers array has 3 elements per entry:
            # 1) audio.  2) user ID. 3) Phone alignments.
            for i, (buffer_, buf_user, buf_align) in enumerate(buffers):
                if self.coord.should_stop():
                    stop = True
                    break

                assert len(buffer_) == len(buf_user) == len(buf_align)

                # Cut samples into fixed size pieces.
                # top up the current buffers[i] element if it
                # is too short.
                while len(buffer_) < self.chunk_size:
                    filename, audio, user, alignment = iterator.next()
                    if self.silence_threshold is not None:
                        # Remove silence
                        audio, user, alignment = \
                            trim_silence(audio, user, alignment,
                                         self.silence_threshold)

                        if audio.size == 0:
                            print("Warning: {} was ignored as it contains "
                                  "only silence. Consider decreasing "
                                  "trim_silence threshold, or adjust volume "
                                  "of the audio.".format(filename))

                    if not self.reverse:
                        buffer_ = np.append(buffer_, audio)
                        buf_user = np.append(buf_user, user)
                        buf_align = np.append(buf_align, alignment)
                    else:
                        buffer_ = np.append(audio, buffer_)
                        buf_user = np.append(user, buf_user)
                        buf_align = np.append(alignment, buf_align)

                # Send one piece
                if not self.reverse:
                    piece = buffer_[:self.chunk_size]
                    piece_user = buf_user[:self.chunk_size]
                    piece_align = buf_align[:self.chunk_size]
                    buffer_ = buffer_[self.chunk_size:]
                    buf_user = buf_user[self.chunk_size:]
                    buf_align = buf_align[self.chunk_size:]
                else:
                    piece = buffer_[-self.chunk_size:]
                    piece_user = buf_user[-self.chunk_size:]
                    piece_align = buf_align[-self.chunk_size:]
                    buffer_ = buffer_[:-self.chunk_size]
                    buf_user = buf_user[:-self.chunk_size]
                    buf_align = buf_align[:-self.chunk_size]
                sess.run(self.enqueue,
                         feed_dict={self.sample_placeholder: piece,
                                    self.user_placeholder: piece_user,
                                    self.align_placeholder: piece_align})
                buffers[i] = (buffer_, buf_user, buf_align)

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
