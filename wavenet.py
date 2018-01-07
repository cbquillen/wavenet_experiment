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


def wavenet_block(xpad, x, conditioning, num_outputs, num_outputs2, rate,
                  is_training, opts, scope):
    '''
    wavenet_block: many important convolution parameters (reuse, kernel_size
    etc.) come from the arg_scope() and are set by wavenet().

    Note that convolutions below end up being causal because the
    input is padded and padding='VALID'. This causes samples to
    drop in the "right" way.
    '''

    dropout, kernel_size, skip_dimension, histogram_summaries = (
        opts.dropout, opts.kernel_size, opts.skip_dimension,
        opts.histogram_summaries)

    conv_gate = layers.conv2d(
        xpad, num_outputs=num_outputs2*2, rate=rate, kernel_size=kernel_size,
        activation_fn=None, normalizer_params=None, scope=scope + '/conv_gate')

    # Add the conditioning.
    conv_gate += layers.conv2d(
        conditioning, num_outputs=num_outputs2*2, rate=rate, kernel_size=1,
        activation_fn=None, normalizer_params=None, scope=scope + '/cur_cond')

    with tf.name_scope(scope + '/activation'):
        conv = tf.nn.tanh(conv_gate[:, :, :num_outputs2], name='conv')
        gate = tf.nn.sigmoid(conv_gate[:, :, num_outputs2:], name='gate')

    with tf.name_scope(scope + '/prod'):
        out = conv * gate

    if dropout > 0:
        out = layers.dropout(out, keep_prob=dropout,
                             is_training=is_training, scope=scope + '/dropout')

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
    The wavenet model definition for training/generation.
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
                'trainable': True,
                'variables_collections': {
                    'moving_mean': [tf.GraphKeys.TRAINABLE_VARIABLES],
                    'moving_variance': [tf.GraphKeys.TRAINABLE_VARIABLES]},
                'reuse': reuse,
                'scale': True,  # Update Variance too.
                'scope': 'BatchNorm',
                # Do updates in place. slower?
                'updates_collections': None,
            }
        }

    l2reg = None
    if 'l2reg' in vars(opts):
        l2reg = tf.contrib.layers.l2_regularizer(opts.l2reg)
    # unpack inputs.
    inputs, user, alignment, lf0 = inputs

    with tf.variable_scope('conditioning'):
        conditioning = tf.one_hot(alignment, depth=opts.n_phones,
                                  name='align_onehot')
        conditioning = tf.reshape(
            conditioning, (opts.n_chunks, -1, opts.n_phones*opts.context))
        if user is not None:
            user = tf.one_hot(user, depth=opts.n_users, name='user_onehot')
            conditioning = tf.concat([user, conditioning], axis=2,
                                     name='cat_user')
        if 'cond_dim' in vars(opts):
            conditioning = layers.conv2d(
                conditioning, num_outputs=opts.cond_dim, kernel_size=(1,),
                rate=1, activation_fn=None, reuse=reuse, scope='cond_vec')
        lf0 = tf.reshape(lf0, (opts.n_chunks, -1, 1))
        conditioning = tf.concat([conditioning, lf0], axis=2, name='cat_lf0')
        if data_format == 'NCW':
            conditioning = tf.transpose(conditioning, [0, 2, 1], name="tr")

    # The arg_scope below will apply to all convolutions, including the ones
    # in wavenet_block().
    with arg_scope([layers.conv2d], data_format=data_format,
                   reuse=reuse, padding='VALID', weights_regularizer=l2reg,
                   **normalizer_params):
        if opts.input_kernel_size > 1:
            inputs = padded(new_x=inputs, reuse=pad_reuse,
                            reverse=opts.reverse, pad=opts.input_kernel_size-1,
                            n_chunks=opts.n_chunks, data_format=data_format,
                            scope='input_layer/pad'+extra_pad_scope)
        x = layers.conv2d(inputs, num_outputs=opts.num_outputs,
                          kernel_size=opts.input_kernel_size, rate=1,
                          activation_fn=tf.nn.tanh, scope='input_layer')

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                block_rate = "block_{}/rate_{}".format(i_block, rate)
                xpad = padded(
                    new_x=x, pad=rate*(opts.kernel_size-1),
                    reuse=pad_reuse, n_chunks=opts.n_chunks,
                    reverse=opts.reverse, data_format=data_format,
                    scope=block_rate+"/pad"+extra_pad_scope)

                x, skip_connection = wavenet_block(
                    xpad, x, conditioning, opts.num_outputs,
                    opts.num_outputs2, rate, is_training,
                    opts, scope=block_rate)

                with tf.name_scope(block_rate+"_skip".format(i_block, rate)):
                    skip_connections += skip_connection

    with tf.name_scope("relu_skip"):
        skip_connections = tf.nn.relu(skip_connections)

    with arg_scope([layers.conv2d], kernel_size=1, reuse=reuse,
                   data_format=data_format):
        x = layers.conv2d(
            skip_connections, num_outputs=opts.skip_dimension,  # ?
            activation_fn=tf.nn.relu, scope='output_layer1')
        mfcc = layers.conv2d(
            x, num_outputs=opts.skip_dimension,   # ?
            activation_fn=tf.nn.relu, scope='mfcc_layer1')
        x = layers.conv2d(
            x, num_outputs=opts.quantization_channels,
            normalizer_params=None,
            activation_fn=None, scope='output_layer2')
        mfcc = layers.conv2d(
            mfcc, num_outputs=opts.n_mfcc, normalizer_params=None,
            activation_fn=None, scope='mfcc_layer2')
    return x, mfcc


def wavenet_unpadded(inputs, opts, is_training=True, reuse=False,
                     data_format=None, extra_pad_scope=''):
    '''
    The wavenet model definition for training/generation.

    Note that this is slower in training than the "padded"
    version, and much slower in generation, especially on the CPU.
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

    overlap = compute_overlap(opts)

    l2reg = None
    if 'l2reg' in vars(opts):
        l2reg = tf.contrib.layers.l2_regularizer(opts.l2reg)
    # unpack inputs.
    inputs, user, alignment, lf0 = inputs

    with tf.variable_scope('conditioning'):
        conditioning = tf.one_hot(alignment, depth=opts.n_phones,
                                  name='align_onehot')
        conditioning = tf.reshape(
            conditioning, (opts.n_chunks, -1, opts.n_phones*opts.context))
        if user is not None:
            user = tf.one_hot(user, depth=opts.n_users, name='user_onehot')
            conditioning = tf.concat([user, conditioning], axis=2,
                                     name='cat_user')
        if 'cond_dim' in vars(opts):
            conditioning = layers.conv2d(
                conditioning, num_outputs=opts.cond_dim, kernel_size=(1,),
                rate=1, activation_fn=None, reuse=reuse, scope='cond_vec')
        lf0 = tf.reshape(lf0, (opts.n_chunks, -1, 1))
        conditioning = tf.concat([conditioning, lf0], axis=2, name='cat_lf0')

    # The arg_scope below will apply to all convolutions, including the ones
    # in wavenet_block().
    with arg_scope([layers.conv2d], data_format=data_format,
                   reuse=reuse, padding='VALID', weights_regularizer=l2reg,
                   **normalizer_params):

        x = layers.conv2d(inputs, num_outputs=opts.num_outputs,
                          kernel_size=opts.input_kernel_size, rate=1,
                          activation_fn=tf.nn.tanh, scope='input_layer')
        cond_offset = opts.input_kernel_size-1

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                block_rate = "block_{}/rate_{}".format(i_block, rate)

                samples_lost = rate*(opts.kernel_size-1)
                cond_offset += samples_lost
                x, skip_connection = wavenet_block(
                    x, x[:, samples_lost:, :],
                    conditioning[:, cond_offset:, :],
                    opts.num_outputs, opts.num_outputs2, rate, is_training,
                    opts, scope=block_rate)

                with tf.name_scope(block_rate+"_skip".format(i_block, rate)):
                    skip_connections += skip_connection[
                        :, overlap-cond_offset:, :]

    with tf.name_scope("relu_skip"):
        skip_connections = tf.nn.relu(skip_connections)

    with arg_scope([layers.conv2d], kernel_size=1, reuse=reuse,
                   data_format=data_format):
        x = layers.conv2d(
            skip_connections, num_outputs=opts.skip_dimension,  # ?
            activation_fn=tf.nn.relu, scope='output_layer1')
        mfcc = layers.conv2d(
            x, num_outputs=opts.skip_dimension,   # ?
            activation_fn=tf.nn.relu, scope='mfcc_layer1')
        x = layers.conv2d(
            x, num_outputs=opts.quantization_channels,
            normalizer_params=None,
            activation_fn=None, scope='output_layer2')
        mfcc = layers.conv2d(
            mfcc, num_outputs=opts.n_mfcc, normalizer_params=None,
            activation_fn=None, scope='mfcc_layer2')
    return x, mfcc


def compute_overlap(opts):
    total_lost = opts.input_kernel_size-1
    for block_dilations in opts.dilations:
        for rate in block_dilations:
            total_lost += rate*(opts.kernel_size-1)
    return total_lost
