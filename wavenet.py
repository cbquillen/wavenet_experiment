#!/usr/bin/env python
'''
Wavenet model definition and generation using contrib.layers
 - Carl Quillen
'''

from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope
from ops import mu_law_decode, mu_law_encode


def wavnet_block(x, num_outputs, rate, kernel_size, skip_dimension,
                 histogram_summaries, scope):
    '''
    wavnet_block: many important convolution parameters (reuse, kernel_size
    etc.) come from the arg_scope() and are set by wavenet().

    Note that convolutions below end up being causal because the
    input is padded and padding='VALID'. This causes samples to
    drop in the "right" way.
    '''

    conv = layers.conv2d(x, num_outputs=num_outputs, rate=rate,
                         kernel_size=kernel_size,
                         activation_fn=tf.nn.tanh,
                         normalizer_params=None,
                         scope=scope + '/conv')

    gate = layers.conv2d(x, num_outputs=num_outputs, rate=rate,
                         kernel_size=kernel_size,
                         activation_fn=tf.nn.sigmoid,
                         normalizer_params=None,
                         scope=scope + '/gate')

    with tf.name_scope(scope + '/prod'):
        out = conv * gate

    out = layers.conv2d(out, num_outputs=num_outputs, kernel_size=1,
                        activation_fn=tf.nn.relu,
                        scope=scope + '/output_xform')

    with tf.name_scope(scope + '/residual'):
        out_sz = out.get_shape()[1].value
        out_sz = out_sz if out_sz is not None else tf.shape(out)[1]
        residual = x[:, -out_sz:, :] + out

    if skip_dimension != num_outputs:      # Upscale for more goodness.
        out = layers.conv2d(out, num_outputs=skip_dimension,
                            kernel_size=1, activation_fn=None,
                            scope=scope + '/skip_upscale')

    if histogram_summaries:
        tf.summary.histogram(name=scope + '/conv', values=conv)
        tf.summary.histogram(name=scope + '/gate', values=gate)
        tf.summary.histogram(name=scope + '/out', values=out)

    return residual, out        # out gets added to the skip connections.


def padded(new_x, pad, scope, reuse=False):
    '''
    Pad new_x, and save the rightmost window for context for the next time
    we do the same convolution.  This context carries across utterances
    during training.  Using this trick also allows us to use the same
    wavenet() routine in training as well as generation.
    '''

    with tf.variable_scope(scope, reuse=reuse):
        x = tf.get_variable('pad', shape=(1, pad, new_x.get_shape()[2]),
                            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                         'padding'],
                            initializer=tf.constant_initializer(),
                            trainable=False)
        y = tf.concat(values=(x, new_x), axis=1)
        x = tf.assign(x, y[:, -pad:, :])
        with tf.get_default_graph().control_dependencies([x]):
            return tf.identity(y)


def wavenet(inputs, opts, is_training=True, reuse=False, pad_reuse=False,
            extra_pad_scope=''):
    '''
    The wavenet model definition for training/generation.  Note that if we
    use wavenets recursively, we will want separate padding variables for
    each "layer".  So we have a separate reuse flag for padding() and
    an additional thing to add to the scope for padding() in that case.
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
    # in wavnet_block().
    with arg_scope([layers.conv2d],
                   reuse=reuse, padding='VALID', **normalizer_params):

        if opts.input_kernel_size > 1:
            inputs = padded(new_x=inputs, reuse=pad_reuse,
                            pad=opts.input_kernel_size-1,
                            scope='input_layer/pad'+extra_pad_scope)
        x = layers.conv2d(
            inputs, num_outputs=opts.num_outputs,
            kernel_size=opts.input_kernel_size, rate=1,
            activation_fn=tf.nn.tanh, scope='input_layer')

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                block_rate = "block_{}/rate_{}".format(i_block, rate)
                x = padded(
                    new_x=x, pad=rate*(opts.kernel_size-1), reuse=pad_reuse,
                    scope=block_rate+"/pad"+extra_pad_scope)

                x, skip_connection = wavnet_block(
                    x, opts.num_outputs, rate, opts.kernel_size,
                    opts.skip_dimension, opts.histogram_summaries,
                    scope=block_rate)

                with tf.name_scope(block_rate+"_skip".format(i_block, rate)):
                    skip_connections += skip_connection

    with tf.name_scope("relu_skip"):
        skip_connections = tf.nn.relu(skip_connections)

    with arg_scope([layers.conv2d], kernel_size=1, reuse=reuse):
        x = layers.conv2d(
            skip_connections, num_outputs=opts.quantization_channels,
            activation_fn=tf.nn.tanh, scope='output_layer1')

        x = layers.conv2d(
            x, num_outputs=opts.quantization_channels,
            normalizer_params=None,
            activation_fn=None, scope='output_layer2')
    return x
