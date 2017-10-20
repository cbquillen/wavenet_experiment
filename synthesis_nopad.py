#!/usr/bin/env python
'''
This is a quick demo hack of a synthesis program.

Carl Quillen
'''

from __future__ import print_function
import optparse
import sys
import time
import os
from operator import mul

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import librosa
from tensorflow.contrib.framework import arg_scope
from ops import mu_law_encode, mu_law_decode
from wavenet import wavenet_unpadded, compute_overlap
from audio_reader import biphone

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
parser.add_option("-c", '--cpu', default=False, dest='use_cpu',
                  action='store_true', help='Set to run on CPU.')

opts, cmdline_args = parser.parse_args()

opts.histogram_summaries = False
opts.reverse = False
opts.silence_phone = 0
opts.user_dim = 10    # User vector dimension to use.
opts.n_phones = 183
opts.n_users = 98
opts.n_mfcc = 12

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
            user_id = np.zeros((1, buf_size), dtype=np.int32) + int(a.pop(0))
            assert a.pop(0) == ':'
            alen = (len(a) - 1)//2
            frame_labels = np.array(map(int, a[0:alen]), dtype=np.int32)
            repeat_factor = sample_rate/100
            read_sample_labels = biphone(frame_labels).repeat(repeat_factor,
                                                              axis=0)
            n_samples = read_sample_labels.shape[0]  # == alen*repeat_factor
            sample_labels = np.zeros(
                (1, n_samples+buf_size-1, 2), dtype=np.int32)
            sample_labels[:, 0:buf_size-1, :] = read_sample_labels[0, :]
            sample_labels[:, buf_size-1:, :] = read_sample_labels[...]
            read_frame_lf0 = np.array(map(float, a[alen+1:]), dtype=np.float32)
            read_frame_lf0 = read_frame_lf0.repeat(repeat_factor)
            sample_lf0 = np.zeros((1, n_samples+buf_size-1), dtype=np.float32)
            sample_lf0[:, 0:buf_size-1] = read_frame_lf0[0]
            sample_lf0[:, buf_size-1:] = read_frame_lf0[...]

            for i in xrange(n_samples):
                yield user_id, sample_labels[:, i:i+buf_size, :], \
                    sample_lf0[:, i:i+buf_size]

buf_size = compute_overlap(opts) + 1
print("Context size is", buf_size-1)
input_dim = opts.quantization_channels if opts.one_hot_input else 1
last_buf = np.zeros((1, buf_size, input_dim), dtype=np.float32)
pLast = tf.placeholder(tf.float32, shape=(1, buf_size, input_dim),
                       name='pLast')
pUser = tf.placeholder(tf.int32, shape=(1, buf_size), name='user')
user = pUser if opts.n_users > 1 else None
pPhone = tf.placeholder(tf.int32, shape=(1, buf_size, 2), name='phone')
pLf0 = tf.placeholder(tf.float32, shape=(1, buf_size), name='lf0')

with tf.name_scope("Generate"):
    # for zeroizing:
    out, _ = wavenet_unpadded([pLast, user, pPhone, pLf0], opts,
                              is_training=False)
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

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

# Run on cpu only.
if opts.use_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # no gpus
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
else:
    sess = tf.Session()
sess.run(init)

print("Restoring from", opts.input_file)
saver.restore(sess, opts.input_file)

samples = []

last_time = time.time()
for iUser, iPhone, iLf0 in align_iterator(opts.input_alignments,
                                          opts.sample_rate):
    output, new_in = sess.run(
        fetches=[max_likeli_sample, out],
        feed_dict={pLast: last_buf, pUser: iUser, pPhone: iPhone,
                   pLf0: iLf0})
    samples.append(output)
    last_buf[:, 0:buf_size-1, :] = last_buf[:, 1:buf_size, :]
    last_buf[:, buf_size-1, :] = new_in[...]
    if len(samples) % 1000 == 999:
        new_time = time.time()
        print("{} samples generated dt={:.02f}".format(len(samples) + 1,
                                                       new_time - last_time))
        last_time = new_time
sess.close()

print("Writing to ", opts.output_file)
librosa.output.write_wav(
    opts.output_file, np.array(samples, dtype=np.float32), opts.sample_rate)