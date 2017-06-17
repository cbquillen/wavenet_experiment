#!/usr/bin/env python
'''
This is a quick demo hack of a synthesis program.

Carl Quillen
'''

from __future__ import print_function
import optparse
import sys
import time
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import librosa
from tensorflow.contrib.framework import arg_scope
from ops import mu_law_encode, mu_law_decode
from wavenet import wavenet

parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-l', '--logdir', dest='logdir',
                  default=None, help='Tensorflow event logdir')
parser.add_option('-i', '--input_file', dest='input_file',
                  default=None, help='Input checkpoint file')
parser.add_option('-a', '--input_alignments', dest='input_alignments',
                  default=None, help='Input alignments file')
parser.add_option('-o', '--output_file', dest='output_file',
                  default='wn_sample.wav', help='Output generated wav file')
parser.add_option('-b', '--batch_norm', dest='batch_norm',
                  action='store_true', default=False,
                  help='Do batch normalization')
parser.add_option("-z", '--zeros', default=16000, dest='initial_zeros',
                  type=int, help='Initial warm-up zero-buffer samples')

opts, cmdline_args = parser.parse_args()

opts.histogram_summaries = False
opts.reverse = False
opts.silence_phone = 0
opts.user_dim = 10    # User vector dimension to use.
opts.n_phones = 183
opts.n_users = 98

# Further options *must* come from a parameter file.
# TODO: add checks that everything is defined.
if opts.param_file is None:
    print("You must provide a parameter file (-p).", file=sys.stderr)
    exit(1)

with open(opts.param_file) as f:
    exec(f)

opts.n_chunks = 1       # force it for wavenet.

if opts.input_file is None:
    print("You must provide an input model (-i).", file=sys.stderr)
    exit(1)

if opts.input_alignments is None:
    print("You must provide an input alignment file (-a).", file=sys.stderr)
    exit(1)


def align_iterator(input_alignments, sample_rate):
    with open(input_alignments) as f:
        for line in f:
            a = line.rstrip().split()
            path = a.pop(0)
            user_id = np.array(int(a.pop(0)), dtype=np.int32).reshape(1, 1)
            assert a.pop(0) == ':'
            frame_labels = np.array(map(int, a), dtype=np.int32)
            frame_labels = frame_labels.repeat(sample_rate/100)
            for i in xrange(frame_labels.shape[0]):
                yield user_id, frame_labels[i:i+1].reshape(1, 1)

input_dim = opts.quantization_channels if opts.one_hot_input else 1
prev_out = np.zeros((1, 1, input_dim), dtype=np.float32)
last_sample = tf.placeholder(tf.float32, shape=(1, 1, input_dim),
                             name='last_sample')
pUser = tf.placeholder(tf.int32, shape=(1, 1), name='user')
pPhone = tf.placeholder(tf.int32, shape=(1, 1), name='phone')

with tf.name_scope("Generate"):
    # In synthesis, we may or may not want to specify the
    # user vector directly.  So leave that out of wavenet().
    user = tf.one_hot(pUser, depth=opts.n_users)
    user = layers.conv2d(user, num_outputs=opts.user_dim,
                         kernel_size=(1,), rate=1, activation_fn=None,
                         reuse=False, scope='user_vec')
    # for zeroizing:
    zuser = tf.zeros((1, opts.initial_zeros), dtype=tf.int32)
    zuser = tf.one_hot(zuser, depth=opts.n_users)
    zuser = layers.conv2d(zuser, num_outputs=opts.user_dim,
                          kernel_size=(1,), rate=1, activation_fn=None,
                          reuse=True, scope='user_vec')

    out = wavenet([last_sample, user, pPhone], opts, is_training=False)
    out = tf.nn.softmax(out)

    max_likeli_sample = tf.reshape(
        mu_law_decode(tf.argmax(out, axis=2), opts.quantization_channels), ())

    # Sample from the output distribution to feed back into the input:
    pick = tf.cumsum(out, axis=2)
    select = tf.random_uniform(shape=())
    x = tf.reduce_sum(tf.cast(pick < select, tf.int32), axis=2)
    if opts.one_hot_input:
        out = tf.one_hot(x, depth=opts.quantization_channels)
    else:
        gen_sample = tf.reshape(
            mu_law_decode(x, opts.quantization_channels), ())
        out = tf.reshape(gen_sample, (1, 1, 1))

saver = tf.train.Saver(tf.trainable_variables())
init = tf.global_variables_initializer()

if opts.initial_zeros > 0:
    with tf.name_scope("Zeroize_state"):
        zalign = tf.constant(opts.silence_phone, shape=(1, opts.initial_zeros),
                             dtype=tf.int32)
        if opts.one_hot_input:
            zero = tf.constant(value=opts.quantization_channels/2,
                               shape=(1, opts.initial_zeros))
            zero = tf.one_hot(zero, depth=opts.quantization_channels)
        else:
            zero = tf.constant(value=0.0, shape=(1, opts.initial_zeros, 1))
        zeroize = wavenet([zero, zuser, zalign], opts,
                          reuse=True, pad_reuse=True, is_training=False)

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

# Run on cpu only.
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
sess.run(init)

print("Restoring from", opts.input_file)
saver.restore(sess, opts.input_file)

if opts.initial_zeros > 0:
    print("Zeroing state")
    sess.run(zeroize)
    print("Starting generation")

samples = []

last_time = time.time()
for iUser, iPhone in align_iterator(opts.input_alignments, opts.sample_rate):
    output, prev_out = sess.run(
        fetches=[max_likeli_sample, out],
        feed_dict={last_sample: prev_out, pUser: iUser, pPhone: iPhone})
    samples.append(output)
    if len(samples) % 1000 == 999:
        new_time = time.time()
        print("{} samples generated dt={:.02f}".format(len(samples) + 1,
                                                       new_time-last_time))
        last_time = new_time
sess.close()

print("Writing to ", opts.output_file)
librosa.output.write_wav(
    opts.output_file, np.array(samples, dtype=np.float32), opts.sample_rate)
