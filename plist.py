#!/usr/bin/python

from __future__ import print_function
import optparse
import sys
import tensorflow as tf
import numpy as np
import re
import os
import librosa
import copy
from wavenet import *
from ops import *

parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-P', '--reverse_params', dest='reverse_params',
                  default=None, help='Parameters for the reversed model')
parser.add_option('-f', '--forward_checkpoint', dest='forward_checkpoint',
                  default=None, help='Input checkpoint file')
parser.add_option('-i', '--input_files', dest='input_files',
                  default=None, help='Input audio file to process')
parser.add_option('-o', '--output_files', dest='output_files',
                  default=None, help='Output audio file')
parser.add_option('-d', '--deterministic', dest='deterministic',
                  action='store_true', default=False,
                  help='Use deterministic sampling in iteration')

opts, cmdline_args = parser.parse_args()
opts.histogram_summaries = False
opts.initial_zeros = 16000

if opts.param_file is None:
    print("Please supply a parameter file.")
    exit(1)

with open(opts.param_file) as f:
    exec(f)

if opts.input_files is None:
    print("Please supply an audio input file list.")
    exit(1)
if opts.output_files is None:
    print("Please supply an audio output file list.")
    exit(1)


def sample(output, quantization_channels, deterministic=False):
    if not deterministic:
        pick = tf.cumsum(tf.nn.softmax(output), axis=2)
        select = tf.random_uniform(shape=(output.get_shape()[0].value,))
        sample = tf.reduce_sum(tf.cast(pick < select, tf.int32), axis=2)
    else:
        sample = tf.argmax(output, axis=2)

    return mu_law_decode(tf.reshape(sample, (1, -1, 1)), quantization_channels)

assert opts.initial_zeros > 0
assert not opts.one_hot_input


in_audio = tf.placeholder(tf.float32, shape=None)
in_audio = tf.reshape(in_audio, (1, -1, 1))

with tf.name_scope("Zeroize_state"):
    if opts.one_hot_input:
        zero = np.zeros((1, opts.initial_zeros,
                         opts.quantization_channels), dtype=np.float32)
        zero[0, :, opts.quantization_channels/2] = 1.0
    else:
        zero = np.zeros((1, opts.initial_zeros, 1), dtype=np.float32)

forward_out, _, _ = wavenet(in_audio, opts, is_training=False)
forward = sample(forward_out, opts.quantization_channels, opts.deterministic)

forward_var_list = []
for v in tf.trainable_variables():
    forward_var_list.append(v)

# sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
sess = tf.Session()

sess.run(tf.global_variables_initializer())

forward_saver = tf.train.Saver(forward_var_list)

tf.get_default_graph().finalize()

print("Loading forward checkpoint")
forward_saver.restore(sess, opts.forward_checkpoint)

with open(opts.input_files) as in_fp:
    with open(opts.output_files) as out_fp:
        for in_file in in_fp:
            print("audio input:", in_file)
            in_file = in_file.rstrip()
            out_file = out_fp.readline().rstrip()
            dir_name = os.path.dirname(out_file)
            if dir_name != "" and not os.path.exists(dir_name):
                os.makedirs(dir_name)

            audio, _ = librosa.load(in_file, sr=opts.sample_rate, mono=True)
            audio = audio.reshape(1, -1, 1)

            print("Zero...")
            sess.run(forward, feed_dict={in_audio: zero})
            print("Forward:")
            audio = sess.run(forward, feed_dict={in_audio: audio})
            print("Saving to ", out_file)
            librosa.output.write_wav(
                out_file, audio.reshape((-1,)), opts.sample_rate)
