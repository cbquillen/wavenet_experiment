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


def wavenet_block(padded_x, x, conditioning, num_outputs, num_outputs2, rate,
                  kernel_size, skip_dimension, histogram_summaries, scope):
    '''
    wavenet_block: many important convolution parameters (reuse, kernel_size
    etc.) come from the arg_scope() and are set by wavenet().

    Note that convolutions below end up being causal because the
    input is padded and padding='VALID'. This causes samples to
    drop in the "right" way.
    '''

    conv = layers.conv2d(
        padded_x, num_outputs=num_outputs2, rate=rate, kernel_size=kernel_size,
        activation_fn=None, normalizer_params=None, scope=scope + '/conv')
    delta = layers.conv2d(
        conditioning, num_outputs=num_outputs2, rate=1, kernel_size=(1,),
        activation_fn=None, normalizer_params=None,
        biases_initializer=None, scope=scope + '/conv_conditioning')
    conv = tf.nn.tanh(conv+delta)

    gate = layers.conv2d(
        padded_x, num_outputs=num_outputs2, rate=rate, kernel_size=kernel_size,
        activation_fn=None, normalizer_params=None, scope=scope + '/gate')
    delta = layers.conv2d(
        conditioning, num_outputs=num_outputs2, rate=1, kernel_size=(1,),
        activation_fn=None, normalizer_params=None,
        biases_initializer=None, scope=scope + '/gate_conditioning')
    gate = tf.nn.sigmoid(gate+delta)

    with tf.name_scope(scope + '/prod'):
        out = conv * gate

    out = layers.conv2d(out, num_outputs=num_outputs, kernel_size=1,
                        activation_fn=None,
                        scope=scope + '/output_xform')

    with tf.name_scope(scope + '/residual'):
        residual = x + out

    if skip_dimension != num_outputs:      # Upscale for more goodness.
        out = layers.conv2d(out, num_outputs=skip_dimension,
                            kernel_size=1, activation_fn=None,
                            scope=scope + '/skip_upscale')

    if histogram_summaries:
        tf.summary.histogram(name=scope + '/conv', values=conv)
        tf.summary.histogram(name=scope + '/gate', values=gate)
        tf.summary.histogram(name=scope + '/out', values=out)

    return residual, out        # out gets added to the skip connections.


def padded(new_x, pad, scope, n_chunks, reuse=False,
           reverse=False, data_format=None):
    '''
    Pad new_x, and save the rightmost window for context for the next time
    we do the same convolution.  This context carries across utterances
    during training.  Using this trick also allows us to use the same
    wavenet() routine in training as well as generation.

    reverse=True for reversing the direction of causality.
    '''

    with tf.variable_scope(scope, reuse=reuse):

        if data_format is 'NCW':
            x = tf.get_variable(
                'pad', shape=(n_chunks, new_x.get_shape()[1], pad),
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'padding'],
                initializer=tf.constant_initializer(), trainable=False)
            if not reverse:
                y = tf.concat(values=(x, new_x), axis=2)
                x = tf.assign(x, y[:, :, -pad:])
            else:
                y = tf.concat(values=(new_x, x), axis=2)
                x = tf.assign(x, y[:, :, :pad])
        else:
            x = tf.get_variable(
                'pad', shape=(n_chunks, pad, new_x.get_shape()[2]),
                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'padding'],
                initializer=tf.constant_initializer(), trainable=False)
            if not reverse:
                y = tf.concat(values=(x, new_x), axis=1)
                x = tf.assign(x, y[:, -pad:, :])
            else:
                y = tf.concat(values=(new_x, x), axis=1)
                x = tf.assign(x, y[:, :pad, :])
        with tf.get_default_graph().control_dependencies([x]):
            return tf.identity(y)


def wavenet(inputs, opts, is_training=True, reuse=False, pad_reuse=False,
            data_format=None, extra_pad_scope=''):
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

    # unpack inputs.
    inputs, user, alignment = inputs

    user = tf.one_hot(user, depth=opts.n_users, name='user_onehot')
    alignment = tf.one_hot(alignment, depth=opts.n_phones,
                           name='align_onehot')
    conditioning = tf.concat([user, alignment], axis=2, name='input_concat')
    if 'cond_dim' in vars(opts):
        conditioning = layers.conv2d(conditioning, num_outputs=opts.cond_dim,
                                     kernel_size=(1,), rate=1,
                                     activation_fn=None,
                                     reuse=reuse, scope='cond_vec')
    if data_format == 'NCW':
        conditioning = tf.transpose(conditioning, [0, 2, 1], name="cond_tr")

    # The arg_scope below will apply to all convolutions, including the ones
    # in wavenet_block().
    with arg_scope([layers.conv2d], data_format=data_format,
                   reuse=reuse, padding='VALID', **normalizer_params):

        delta = layers.conv2d(conditioning, num_outputs=opts.num_outputs,
                              kernel_size=(1,), rate=1, activation_fn=None,
                              biases_initializer=None,
                              reuse=reuse, scope='input_conditioning')
        if opts.input_kernel_size > 1:
            inputs = padded(new_x=inputs, reuse=pad_reuse,
                            reverse=opts.reverse, pad=opts.input_kernel_size-1,
                            n_chunks=opts.n_chunks, data_format=data_format,
                            scope='input_layer/pad'+extra_pad_scope)
        x = layers.conv2d(inputs, num_outputs=opts.num_outputs,
                          kernel_size=opts.input_kernel_size, rate=1,
                          activation_fn=None, scope='input_layer')
        x = tf.nn.tanh(x + delta)

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                block_rate = "block_{}/rate_{}".format(i_block, rate)
                padded_x = padded(
                    new_x=x, pad=rate*(opts.kernel_size-1),
                    reuse=pad_reuse, n_chunks=opts.n_chunks,
                    reverse=opts.reverse, data_format=data_format,
                    scope=block_rate+"/pad"+extra_pad_scope)

                x, skip_connection = wavenet_block(
                    padded_x, x, conditioning, opts.num_outputs,
                    opts.num_outputs2, rate, opts.kernel_size,
                    opts.skip_dimension, opts.histogram_summaries,
                    scope=block_rate)

                with tf.name_scope(block_rate+"_skip".format(i_block, rate)):
                    skip_connections += skip_connection

    with tf.name_scope("relu_skip"):
        skip_connections = tf.nn.relu(skip_connections)

    with arg_scope([layers.conv2d], kernel_size=1, reuse=reuse,
                   data_format=data_format):
        x = layers.conv2d(
            skip_connections, num_outputs=opts.quantization_channels,  # ?
            activation_fn=tf.nn.relu, scope='output_layer1')
        mfcc = layers.conv2d(
            x, num_outputs=opts.quantization_channels,   # ?
            activation_fn=tf.nn.relu, scope='mfcc_layer1')
        x = layers.conv2d(
            x, num_outputs=opts.quantization_channels,
            normalizer_params=None,
            activation_fn=None, scope='output_layer2')
        mfcc = layers.conv2d(
            mfcc, num_outputs=1, normalizer_params=None,
            activation_fn=None, scope='mfcc_layer2')
        mfcc = tf.reshape(mfcc, (opts.n_chunks, -1,))
    return x, mfcc
