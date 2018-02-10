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
from wavenet import wavenet, compute_overlap

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
parser.add_option("-c", '--cpu', default=False, dest='use_cpu',
                  action='store_true', help='Set to run on CPU.')
parser.add_option('-n', '--noise_sample', dest='noise_sample',
                  action='store_true', help='Output samples with noise')

opts, cmdline_args = parser.parse_args()

opts.histogram_summaries = False
opts.reverse = False
opts.silence_phone = 0
opts.n_phones = 41
opts.n_users = 1
opts.context = 3      # 2 == biphone, 3 == triphone
opts.n_mfcc = 20
opts.sample_skip = 0.01       # Uniform sample from (this, 1-this).

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


def align_iterator(input_alignments, sample_rate, context):
    with open(input_alignments) as f:
        for line in f:
            a = line.rstrip().split()
            path = a.pop(0)
            user_id = np.array(int(a.pop(0)), dtype=np.int32).reshape(1, 1)
            assert a.pop(0) == ':'
            alen = (len(a) - 1)//(context+1)
            assert a[alen*context] == ':'
            frame_labels = np.array(map(int, a[0:alen*context]),
                                    dtype=np.int32)
            frame_labels = frame_labels.reshape(-1, context)
            frame_lf0 = np.array(map(float, a[alen*context+1:]),
                                 dtype=np.float32)
            repeat_factor = sample_rate/100
            for i in xrange(frame_labels.shape[0]):
                for j in xrange(repeat_factor):
                    yield (user_id, frame_labels[i:i+1, :],
                           frame_lf0[i:i+1].reshape(1, 1))

input_dim = 1
prev_out = np.zeros((1, 1, input_dim), dtype=np.float32)
last_sample = tf.placeholder(tf.float32, shape=(1, 1, input_dim),
                             name='last_sample')
pUser = tf.placeholder(tf.int32, shape=(1, 1), name='user')
user = pUser if opts.n_users > 1 else None
pPhone = tf.placeholder(tf.int32, shape=(1, opts.context), name='phone')
pLf0 = tf.placeholder(tf.float32, shape=(1, 1), name='lf0')

with tf.name_scope("Generate"):
    # for zeroizing:
    mu, r, q, _ = wavenet([last_sample, user, pPhone, pLf0], opts,
                          is_training=False)
    x = tf.random_uniform(
        tf.shape(r), dtype=tf.float32, minval=opts.sample_skip,
        maxval=1.0-opts.sample_skip)
    s1 = 1.0/(r+q)*tf.log(x*2*r/(r - q))
    s2 = 1.0/(q-r)*tf.log((1-x)*2*r/(r + q))
    thresh = 0.5*(r-q)/r
    sample = 0.5*(s1 + s2 + tf.sign(x-thresh)*(s2-s1)) + mu
    sample = tf.expand_dims(sample, -1)
    clipped_mu = tf.clip_by_value(mu, -1.0, 1.0)

saver = tf.train.Saver(tf.trainable_variables() +
                       tf.get_collection('batch_norm'))
init = tf.global_variables_initializer()

initial_zeros = compute_overlap(opts)
with tf.name_scope("Zeroize_state"):
    zuser = None if opts.n_users <= 1 else tf.zeros(
        (1, initial_zeros), dtype=tf.int32)
    zalign = tf.constant(opts.silence_phone, dtype=tf.int32,
                         shape=(1, initial_zeros, opts.context))
    zLf0 = tf.zeros((1, initial_zeros), dtype=tf.float32)
    zero = tf.constant(value=0.0, shape=(1, initial_zeros, 1))
    zeroize = wavenet([zero, zuser, zalign, zLf0], opts,
                      reuse=True, pad_reuse=True, is_training=False)[0]

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
print("Zeroizing state")
sess.run(zeroize)
print("Starting generation")

samples = []

last_time = time.time()
for iUser, iPhone, iLf0 in align_iterator(opts.input_alignments,
                                          opts.sample_rate, opts.context):
    prev_out, max_likeli_sample = sess.run(
        fetches=[sample, clipped_mu],
        feed_dict={last_sample: prev_out, pUser: iUser, pPhone: iPhone,
                   pLf0: iLf0})
    if opts.noise_sample:
        samples.append(prev_out[0, 0, 0])
    else:
        samples.append(max_likeli_sample[0, 0])
    if len(samples) % 1000 == 999:
        new_time = time.time()
        print("{} samples generated dt={:.02f}".format(len(samples) + 1,
                                                       new_time - last_time))
        last_time = new_time
sess.close()

print("Writing to ", opts.output_file)
librosa.output.write_wav(
    opts.output_file, np.array(samples, dtype=np.float32), opts.sample_rate)
