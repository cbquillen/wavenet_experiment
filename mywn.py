#!/usr/bin/python
'''
My first stab at a wavenet trainer.
'''

import optparse
import sys
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from audio_reader import AudioReader
from ops import causal_atrous_conv1d

parser = optparse.OptionParser()
parser.add_option('-l', '--logdir', dest='logdir',
                  default=None, help='Tensorflow event logdir')
parser.add_option('-d', '--data', dest='data_dir',
                  default='VCTK-Corpus', help='VCTK data directory')
parser.add_option('-s', '--sample_rate', dest='sample_rate',
                  default=16000, help='Sample rate')
parser.add_option('-L', '--learning_rate', dest='learning_rate',
                  default=1e-05, help='Initial learning rate')
parser.add_option('-S', '--silence_threshold', dest='silence_threshold',
                  default=0.0, help='Silence classifier energy threshold')

opts, cmdline_args = parser.parse_args()

coord = tf.train.Coordinator()
data = AudioReader(opts.data_dir, coord, opts.sample_rate, sample_size=None,
                   silence_threshold=opts.silence_threshold, queue_size=256)

sess = tf.Session()
data.start_threads(sess)

def resnet_block(x, n_outputs, rate, reuse, scope):
    block_scope = scope+'/'+str(rate)
    conv = causal_atrous_conv1d(x, num_outputs = num_outputs, rate = rate,
    	activation_fn = tf.nn.tanh, reuse = reuse, scope = block_scope+'/conv')
    gate = causal_atrous_conv1d(x, num_outputs = num_outputs, rate = rate,
    	activation_fn = tf.nn.sigmoid, reuse = reuse, scope = block_scope+'/gate')
    with tf.name_scope(block_scope+'/prod'):
	out = conv*gate
    out = layers.conv1d(out, num_outputs = num_outputs, kernel_size = 1,
    	activation_fn = tf.nn.tanh, reuse = reuse, scope = block_scope+'/output_xform')
    with tf.name_scope(block_scope+'/residual'):
	residual = x + out
    return residual, out

def wavenet(inputs, num_outputs, num_blocks, dilations_per_block, kernel_size,
	reuse = True):
    x = tf.reshape(inputs, [1, -1, 1])
    skip_connections = 0
    for n in range num_blocks:
	for rate in dilations_per_block:
	    with arg_scope([causal_atrous_conv1d], kernel_size = kernel_size):
		skip_connection, x = resnet_block(x, num_outputs, rate,
			reuse, scope='block_{}'.format(n))
		skip_connections += skip_connection
 
