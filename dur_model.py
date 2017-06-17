#!/usr/bin/env python
'''
A quite bogus LSTM duration model

Carl Quillen
'''

from __future__ import print_function

import optparse
import sys
import time
import math
import random
import threading
import numpy as np
from operator import mul

import tensorflow as tf
import tensorflow.contrib.layers as layers

# Options from the command line:
parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-l', '--logdir', dest='logdir',
                  default=None, help='Tensorflow event logdir')
parser.add_option('-i', '--input_file', dest='input_file',
                  default=None, help='Input checkpoint file')
parser.add_option('-o', '--output_file', dest='output_file',
                  default='ckpt', help='Output checkpoint file')
parser.add_option('-d', '--data_file', dest='data_file',
                  default='durs.txt', help='VCTK corpus duration file')
parser.add_option('-m', '--phone_map_file', dest='phone_map_file',
                  default='phone.map', help='VCTK phone map.')
parser.add_option('-c', '--checkpoint_rate', dest='checkpoint_rate',
                  type=int, default=10000, help='Rate to checkpoint.')
parser.add_option('-s', '--summary_rate', dest='summary_rate',
                  type=int, default=20, help='Rate to output summaries.')
parser.add_option('-L', '--base_learning_rate', dest='base_learning_rate',
                  type=float, default=1e-03,
                  help='The initial learning rate. ' +
                  'lr = base_learning_rate/(1.0+lr_offet+timestep)*const)')
parser.add_option('-O', '--lr_offset', dest='lr_offset', type=int, default=0,
                  help="lr=base_learning_rate/(1.0+timestep+lr_offset)*const)")

opts, cmdline_args = parser.parse_args()

# Options that can be set in a parameter file:
opts.canonical_epoch_size = 2000.0
opts.num_units = 128
opts.chunk_size = 50
opts.n_chunks = 32
opts.max_checkpoints = 100
opts.clip = None                # Derivative clipping
opts.epsilon = 1e-4             # Adam optimizer epsilon.
opts.max_steps = 7000           # About 10 epochs.
opts.max_silence = 20
opts.silence_phone = 0

# Set opts.* parameters from a parameter file if you want:
if opts.param_file is not None:
    with open(opts.param_file) as f:
        exec(f)

sess = tf.Session()


# Never finishes.
def align_iterator(data_file):
    epoch = 0
    alignments = []
    with open(data_file) as data:
        for line in data:
            plist = []
            durlist = []
            els = line.rstrip().split()
            for el in els:
                phone, dur = map(int, el.split(':'))
                # Clip silence duration:
                if phone == opts.silence_phone and dur > opts.max_silence:
                    dur = opts.max_silence
                plist.append(phone)
                durlist.append(dur)
            phone_array = np.array(plist, dtype=np.int32)
            # make the plist a two element array, the current phone and
            # the next through some hackery.
            phone_array = np.append(phone_array.repeat(2), phone_array[-1])
            phone_array = phone_array[1:].reshape(-1, 2)
            alignments.append((phone_array,
                               np.array(durlist, dtype=np.int32)))

    while True:
        random.shuffle(alignments)
        for phones, durs in alignments:
            yield phones, durs

        print("Epoch {} ended".format(epoch))
        epoch += 1


class AlignReader(object):
    '''Generic background alignment reader that preprocesses alignements
    and enqueues them into a TensorFlow queue.'''

    def __init__(self, alignment_list_file, chunk_size, n_chunks=5,
                 queue_size=5):
        self.iterator = align_iterator(alignment_list_file)
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.threads = []
        self.pPhones = tf.placeholder(dtype=tf.int32, shape=(chunk_size, 2))
        self.pDurs = tf.placeholder(dtype=tf.int32, shape=(chunk_size,))
        self.queue = tf.PaddingFIFOQueue(
            queue_size, ['int32', 'int32'],
            shapes=[(chunk_size, 2), (chunk_size,)])
        self.enqueue = self.queue.enqueue([self.pPhones, self.pDurs])

    def dequeue(self, num_elements):
        return self.queue.dequeue_many(num_elements)

    # Thread main is a little tricky.  We want to enqueue multiple chunks,
    # each from a separate utterance.
    # We keep an array of buffers for this.  We cut fixed sized chunks
    # out of the buffers.  As each buffer exhausts, we load a new
    # alignment file (using align_iterator) and concatenate it with the
    # buffer remnants.
    def thread_main(self, sess):
        # buffers: the array of buffers.
        buffers = [(np.array([], dtype=np.int32).reshape(0, 2),
                    np.array([], dtype=np.int32))]*self.n_chunks

        stop = False
        while not stop:
            # The buffers array has 2 elements per entry:
            # 1) phones  2) phone durations
            for i, (buf_phones, buf_durs) in enumerate(buffers):
                # Don't bother with a coordinator yet...
                # if self.coord.should_stop():
                #    stop = True
                #    break

                assert len(buf_phones) == len(buf_durs)

                # Cut alignments into fixed size pieces.
                # top up the current buffers[i] element if it
                # is too short.
                while len(buf_phones) < self.chunk_size:
                    # iterator.next() will never finish.  It will allow us to
                    # go through the data set multiple times.
                    phones, durs = self.iterator.next()

                    buf_phones = np.append(buf_phones, phones, axis=0)
                    buf_durs = np.append(buf_durs, durs)
                    assert buf_phones.shape[0] == buf_durs.shape[0]

                # Send one piece
                piece_phones = buf_phones[:self.chunk_size, :]
                piece_durs = buf_durs[:self.chunk_size]
                buf_phones = buf_phones[self.chunk_size:, :]
                buf_durs = buf_durs[self.chunk_size:]

                sess.run(self.enqueue,
                         feed_dict={self.pPhones: piece_phones,
                                    self.pDurs: piece_durs})
                buffers[i] = (buf_phones, buf_durs)

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

align_reader = AlignReader(opts.data_file, chunk_size=opts.chunk_size,
                           n_chunks=opts.n_chunks, queue_size=opts.n_chunks)

phone_map = {}
with open(opts.phone_map_file) as p:
    for line in p:
        val, phone = line.rstrip().split()
        phone_map[phone] = int(val)
n_phones = len(phone_map)

cell = tf.contrib.rnn.LSTMCell(num_units=opts.num_units, num_proj=1)
zero_state = cell.zero_state(opts.n_chunks, tf.float32)
state_list = []
for i, s in enumerate(cell.zero_state(opts.n_chunks, tf.float32)):
    state_list.append(
        tf.get_variable(initializer=s, name="cur_state_"+str(i)))

with tf.name_scope("input_massaging"):
    [phones, durs] = align_reader.dequeue(opts.n_chunks)
    phone_vec = tf.one_hot(phones, depth=n_phones, axis=-1)
    phone_vec = tf.reshape(phone_vec, (opts.n_chunks, -1, n_phones*2))
    dur_vec = tf.reshape(tf.cast(durs, tf.float32), (opts.n_chunks, -1, 1))

# Define the computational graph.
output, state = tf.nn.dynamic_rnn(
    cell=cell, initial_state=tf.contrib.rnn.LSTMStateTuple(*state_list),
    inputs=phone_vec)

assert len(state_list) == len(state)
state_updates = []
for i, state_var in enumerate(state_list):
    state_updates.append(tf.assign(state_var, state[i]))
with tf.get_default_graph().control_dependencies(state_updates):
    loss = tf.norm(output - dur_vec, ord=1)/opts.chunk_size/opts.n_chunks

# That should have created all training variables.  Now we can make a saver.
saver = tf.train.Saver(tf.trainable_variables(),
                       max_to_keep=opts.max_checkpoints)

tf.summary.scalar(name="loss", tensor=loss)

learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
# adams_epsilon probably should be reduced near the end of training.
adams_epsilon = tf.placeholder(tf.float32, shape=(), name='adams_epsilon')

# We might want to run just measuring loss and not training,
# perhaps to see what the loss variance is on the training.
# in that case, set opts.base_learning_rate=0
if opts.base_learning_rate > 0:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       epsilon=adams_epsilon)
    if opts.clip is not None:
        gradients = optimizer.compute_gradients(
            loss, var_list=tf.trainable_variables())
        clipped_gradients = [(tf.clip_by_value(var, -opts.clip, opts.clip),
                              name) for var, name in gradients]
        minimize = optimizer.apply_gradients(clipped_gradients)
    else:
        minimize = optimizer.minimize(loss, var_list=tf.trainable_variables())
else:
    minimize = tf.constant(0)   # a noop.

summaries = tf.summary.merge_all()

init = tf.global_variables_initializer()

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

print("Model variables:")
total_params = 0
for var in tf.trainable_variables():
    vshape = var.get_shape().as_list()
    total_params += reduce(mul, vshape)
    print("  ", var.name, vshape)
print("Total model parameters:", total_params)
sys.stdout.flush()

if opts.logdir is not None:
    summary_writer = tf.summary.FileWriter(logdir=opts.logdir,
                                           graph=tf.get_default_graph())

# Initialize everything.
sess.run(init)

if opts.input_file is not None:
    print("Restoring from", opts.input_file)
    saver.restore(sess, opts.input_file)

# Main training loop:
align_reader.start_threads(sess)

last_time = time.time()
data_index = 0

for global_step in xrange(opts.lr_offset, opts.max_steps):
    cur_lr = opts.base_learning_rate/(
        1.0 + global_step/opts.canonical_epoch_size)

    if (global_step + 1) % opts.summary_rate == 0 and opts.logdir is not None:
        cur_loss, summary_pb = sess.run(
            [loss, summaries, minimize],
            feed_dict={learning_rate: cur_lr,
                       adams_epsilon: opts.epsilon})[0:2]
        summary_writer.add_summary(summary_pb, global_step)
    else:
        cur_loss = sess.run(
            [loss, state, minimize],
            feed_dict={learning_rate: cur_lr,
                       adams_epsilon: opts.epsilon})[0]
    new_time = time.time()
    print("loss[{}]: {:.3f} dt {:.3f} lr {:.4g}".format(
        global_step, cur_loss, new_time - last_time, cur_lr))
    last_time = new_time

    if (global_step + 1) % opts.checkpoint_rate == 0 and \
            opts.output_file is not None:
        saver.save(sess, opts.output_file, global_step)

    sys.stdout.flush()

print("Training done.")
if opts.output_file is not None:
    saver.save(sess, opts.output_file)
sess.close()
