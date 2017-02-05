#!/usr/bin/env python
'''
This is a quick demo hack of a wavenet trainer, based on
some machinery from ibab.

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
from wavenet import wavenet_gen

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

generate = wavenet_gen(opts)

saver = tf.train.Saver(tf.trainable_variables())
init = tf.global_variables_initializer()

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

sess = tf.Session()
sess.run(init)

print("Restoring from", opts.input_file)
saver.restore(sess, opts.input_file)

output = np.zeros((opts.num_samples), dtype=np.float32)

for sample in xrange(opts.num_samples):
    output[sample] = sess.run(generate)
    if sample % 1000 == 999:
        print("{} samples generated.".format(sample + 1))
sess.close()

librosa.output.write_wav(opts.filename, opts.output_file,
                         opts.sample_rate)
