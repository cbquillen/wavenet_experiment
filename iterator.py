#!/usr/bin/python

from __future__ import print_function
import optparse
import sys
import tensorflow as tf
import numpy as np
import re
import librosa
import copy
from wavenet import *
from ops import *


def sample(output, quantization_channels):
    pick = tf.cumsum(tf.nn.softmax(output), axis=2)
    select = tf.random_uniform(shape=(output.get_shape()[0].value,))
    sample = tf.reduce_sum(tf.cast(pick < select, tf.int32), axis=2)
    return mu_law_decode(tf.reshape(sample, (1, -1, 1)), quantization_channels)

parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-P', '--reverse_params', dest='reverse_params',
                  default=None, help='Parameters for the reversed model')
parser.add_option('-f', '--forward_checkpoint', dest='forward_checkpoint',
                  default=None, help='Input checkpoint file')
parser.add_option('-r', '--reverse_checkpoint', dest='reverse_checkpoint',
                  default=None,
                  help='Input checkpoint file for reversed direction')
parser.add_option('-i', '--input_file', dest='input_file',
                  default=None, help='Input audio file to process')
parser.add_option('-o', '--output_file', dest='output_file',
                  default=None, help='Output audio file')
parser.add_option('-n', '--iterations', dest='n_iterations',
                  type=int, default=10, help='Number of iterations')


opts, cmdline_args = parser.parse_args()
opts.histogram_summaries = False

if opts.param_file is None:
    print("Please supply a parameter file.")
    exit(1)

with open(opts.param_file) as f:
    exec(f)

if opts.input_file is None:
    print("Please supply an audio input file.")
    exit(1)

audio, _ = librosa.load(opts.input_file, sr=opts.sample_rate, mono=True)
audio = audio.reshape(1, -1, 1)

in_audio = tf.placeholder(tf.float32, shape=audio.shape)

forward_out = wavenet(in_audio, opts, is_training=False)
forward = sample(forward_out, opts.quantization_channels)

forward_var_list = []
for v in tf.trainable_variables():
    forward_var_list.append(v)

if opts.reverse_params is None:
    print("Please supply a parameter file.")

forward_opts = copy.copy(opts)
with open(opts.reverse_params) as f:
    exec(f)

reverse_opts = opts
assert not reverse_opts.one_hot_input      # I don't implement it yet.

with tf.variable_scope('reverse'):
    reverse = wavenet(forward, reverse_opts, is_training=False)
    reverse = sample(reverse, reverse_opts.quantization_channels)

sess = tf.Session()

reverse_var_map = {}
for v in tf.trainable_variables():
    if v.name.startswith('reverse/'):
        new_name = re.sub(r'^reverse/', '', v.name)
        new_name = re.sub(r':0$', '', new_name)
        reverse_var_map[new_name] = v

sess.run(tf.global_variables_initializer())

forward_saver = tf.train.Saver(forward_var_list)
reverse_saver = tf.train.Saver(reverse_var_map)

tf.get_default_graph().finalize()

print("Loading forward checkpoint")
forward_saver.restore(sess, opts.forward_checkpoint)

print("Loading reverse checkpoint")
reverse_saver.restore(sess, opts.reverse_checkpoint)

for i in xrange(opts.n_iterations):
    forward_out, audio = sess.run([forward, reverse], feed_dict={in_audio: audio})
    print("Saving to ", str(i) + '_' + opts.output_file)
    librosa.output.write_wav(
        str(i) + '_' + opts.output_file, audio.reshape((-1,)),
        opts.sample_rate)
    librosa.output.write_wav(
        'f'+str(i) + '_' + opts.output_file, forward_out.reshape((-1,)),
        opts.sample_rate)
