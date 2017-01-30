#!/usr/bin/env python
# Wavenet model definition using contrib.layers
# - Carl Quillen

import tensorflow as tf
import tensorflow.contrib.layers as layers
from ops import causal_atrous_conv1d
from tensorflow.contrib.framework import arg_scope, add_arg_scope


def resnet_block(x, num_outputs, rate, atrous_kernel_size,
                 skip_dimension, histogram_summaries, scope):
    '''
    resnet_block: many important convolution parameters (reuse, kernel_size
    etc.) come from the arg_scope() and are set by wavenet().
    '''
    block_scope = scope + '/rate_' + str(rate)

    conv = causal_atrous_conv1d(x, num_outputs=num_outputs, rate=rate,
                                kernel_size=atrous_kernel_size,
                                activation_fn=tf.nn.tanh,
                                scope=block_scope + '/conv')

    gate = causal_atrous_conv1d(x, num_outputs=num_outputs, rate=rate,
                                kernel_size=atrous_kernel_size,
                                activation_fn=tf.nn.sigmoid,
                                scope=block_scope + '/gate')

    with tf.name_scope(block_scope + '/prod'):
        out = conv * gate

    out = layers.convolution(out, num_outputs=num_outputs, kernel_size=1,
                             activation_fn=tf.nn.tanh,
                             scope=block_scope + '/output_xform')

    with tf.name_scope(block_scope + '/residual'):
        residual = x + out

    if skip_dimension != num_outputs:      # Upscale for more goodness.
        out = layers.convolution(out, num_outputs=skip_dimension,
                                 kernel_size=1, activation_fn=None,
                                 scope=block_scope + '/skip_upscale')

    if histogram_summaries:
        tf.summary.histogram(name=block_scope + '/conv', values=conv)
        tf.summary.histogram(name=block_scope + '/gate', values=gate)
        tf.summary.histogram(name=block_scope + '/out', values=out)

    return residual, out        # out gets added to the skip connections.


def wavenet(inputs, opts, is_training=True, reuse=False):
    '''
    The wavenet model definition for training/testing.
    Generation will be elsewhere.
    '''

    # Parameters for batch normalization
    normalizer_params = {}
    if opts.batch_norm:
        normalizer_params = {
            'normalizer_fn': layers.batch_norm,
            'normalizer_params': {
                'is_training': is_training,
                'reuse': reuse,
                # Do updates in place. slower?
                'updates_collections': None,
            }
        }

    # The arg_scope below will apply to all convolutions, including the ones
    # in resnet_block().
    with arg_scope([causal_atrous_conv1d, layers.convolution],
                   reuse=reuse, padding='SAME', **normalizer_params):

        x = causal_atrous_conv1d(inputs, num_outputs=opts.num_outputs,
                                 kernel_size=opts.input_kernel_size, rate=1,
                                 activation_fn=tf.nn.tanh, scope='input')

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                x, skip_connection = \
                    resnet_block(x, opts.num_outputs, rate,
                                 opts.kernel_size, opts.skip_dimension,
                                 opts.histogram_summaries,
                                 scope='block_{}'.format(i_block))

                with tf.name_scope(
                        "block_{}/rate_{}_skip".format(i_block, rate)):
                    skip_connections += skip_connection

        with arg_scope([layers.convolution], kernel_size=1):
            x = layers.convolution(skip_connections,
                                   num_outputs=opts.quantization_channels,
                                   activation_fn=tf.nn.tanh,
                                   scope='output_layer1')
            x = layers.convolution(x, num_outputs=opts.quantization_channels,
                                   activation_fn=None, scope='output_layer2')
            return x
