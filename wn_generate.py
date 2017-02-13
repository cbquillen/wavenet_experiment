#!/usr/bin/env python
'''
This is a quick demo hack of a wavenet generator.

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
parser.add_option('-o', '--output_file', dest='output_file',
                  default='wn_sample.wav', help='Output generated wav file')
parser.add_option('-Z', '--audio_chunk_size', dest='audio_chunk_size',
                  type=int, default=100000, help='Audio chunk size per batch.')
parser.add_option('-H', '--histogram_summaries', dest='histogram_summaries',
                  action='store_true', default=False,
                  help='Do histogram summaries')
parser.add_option('-b', '--batch_norm', dest='batch_norm',
                  action='store_true', default=False,
                  help='Do batch normalization')
parser.add_option('-n', '--num_samples', default=32000, dest='num_samples',
                  type=int, help='Samples to generate')

opts, cmdline_args = parser.parse_args()

# Further options *must( come from a parameter file.
# TODO: add checks that everything is defined.

# Set opts.* parameters from a parameter file if you want:
if opts.param_file is None:
    print("You must provide a parameter file (-p).", file=sys.stderr)
    exit(1)

with open(opts.param_file) as f:
    exec(f)

if opts.input_file is None:
    print("You must provide an input model (-i).", file=sys.stderr)
    exit(1)

input_dim = opts.quantization_channels if opts.one_hot_input else 1
prev_out = np.zeros((1, 1, input_dim), dtype=np.float32)
last_sample = tf.placeholder(tf.float32, shape=(1, 1, input_dim),
                             name='last_sample')

with tf.name_scope("Generate"):
    out = wavenet(last_sample, opts, is_training=False)
    x = tf.arg_max(out, dimension=2)
    gen_sample = tf.reshape(mu_law_decode(x, opts.quantization_channels), ())
    if not opts.one_hot_input:
        out = tf.reshape(gen_sample, (1, 1, 1))

saver = tf.train.Saver(tf.trainable_variables())
init = tf.global_variables_initializer()

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

# Run on cpu only.
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
sess.run(init)

print("Restoring from", opts.input_file)
saver.restore(sess, opts.input_file)

output = np.zeros((opts.num_samples), dtype=np.float32)

for sample in xrange(opts.num_samples):
    output[sample], prev_out = sess.run(
        fetches=[gen_sample, out], feed_dict={last_sample: prev_out})
    prev_out += (np.random.random()-0.5)*0.0003
    if sample % 1000 == 999:
        print("{} samples generated.".format(sample + 1))
sess.close()

print("Writing to ", opts.output_file)
librosa.output.write_wav(opts.output_file, output, opts.sample_rate)
