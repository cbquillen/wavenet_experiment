#!/usr/bin/env python
# Wavenet model definition and generation using contrib.layers
# - Carl Quillen

import tensorflow as tf
import tensorflow.contrib.layers as layers
from ops import causal_atrous_conv1d
from tensorflow.contrib.framework import arg_scope, add_arg_scope


def wavnet_block(x, num_outputs, rate, kernel_size, skip_dimension,
                 convolution_func, histogram_summaries, scope):
    '''
    wavnet_block: many important convolution parameters (reuse, kernel_size
    etc.) come from the arg_scope() and are set by wavenet().
    '''
    block_scope = scope + '/rate_' + str(rate)

    conv = convolution_func(x, num_outputs=num_outputs, rate=rate,
                         kernel_size=kernel_size,
                         activation_fn=tf.nn.tanh,
                         scope=block_scope + '/conv')

    gate = convolution_func(x, num_outputs=num_outputs, rate=rate,
                         kernel_size=kernel_size,
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
    # in wavnet_block().
    with arg_scope([causal_atrous_conv1d, layers.convolution],
                   reuse=reuse, padding='SAME', **normalizer_params):

        x = causal_atrous_conv1d(inputs, num_outputs=opts.num_outputs,
                                 kernel_size=opts.input_kernel_size, rate=1,
                                 activation_fn=tf.nn.tanh, scope='input')

        skip_connections = 0
        for i_block, block_dilations in enumerate(opts.dilations):
            for rate in block_dilations:
                x, skip_connection = wavnet_block(
                        x, opts.num_outputs, rate, opts.kernel_size,
                        opts.skip_dimension, causal_atrous_conv1d,
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


# A default variable initializer...
def _initializer(shape, dtype, partition_info=None):
    return tf.random_normal(shape, dtype=dtype)*0.1


# Get a variable, shift left and assign a new value at -1.
def shifted_var(name, shape, new_x):
    assert new_x.get_shape()[1] == 1

    with tf.variable_scope("shift_assign"):
        x = tf.get_variable(name=name, shape=shape, trainable=False)
        y = tf.concat(1, [x[:, 1:, :], new_x])
        return tf.assign(x, y)


# wavenet_gen(): Used to propagate generation one sample at a time.
# Almost the same as wavnet()
# ...this isn't going to be fast...
def wavenet_gen(opts):

    # Parameters for batch normalization
    normalizer_params = {}
    if opts.batch_norm:
        normalizer_params = {
            'normalizer_fn': layers.batch_norm,
            'normalizer_params': {
                'is_training': False
            }
        }

    # The arg_scope below will apply to all convolutions, including the ones
    # in wavnet_block().
    with arg_scope([layers.convolution],
                   reuse=False, padding='VALID', **normalizer_params):
        with tf.variable_scope('generate', dtype=tf.float32,
                               initializer=_initializer, reuse=False):
            inputs = tf.get_variable(
                name="input", trainable=False,
                shape=(1, opts.input_kernel_size, opts.num_outputs))

            last_x = layers.convolution(
                inputs, num_outputs=opts.num_outputs,
                kernel_size=opts.input_kernel_size, rate=1,
                activation_fn=tf.nn.tanh, scope='input')

            skip_connections = 0
            for i_block, block_dilations in enumerate(opts.dilations):
                for rate in block_dilations:
                    print "block {} rate {}".format(i_block, rate)
                    x = shifted_var(name="x", new_x=last_x, shape=(
                        1, rate*opts.kernel_size-1, opts.num_outputs))

                    last_x, skip_connection = wavnet_block(
                            x, opts.num_outputs, rate, opts.kernel_size,
                            opts.skip_dimension, layers.convolution,
                            opts.histogram_summaries,
                            scope='block_{}'.format(i_block))

                with tf.name_scope(
                        "block_{}/rate_{}_skip".format(i_block, rate)):
                    skip_connections += skip_connection

        with arg_scope([layers.convolution], kernel_size=1):
            with tf.variable_scope('output_layers'):
                x = layers.convolution(
                    skip_connections, num_outputs=opts.quantization_channels,
                    activation_fn=tf.nn.tanh, scope='output_layer1')

                x = layers.convolution(
                    x, num_outputs=opts.quantization_channels,
                    activation_fn=None, scope='output_layer2')
                x = mu_law_decode(x, quantization_channels)

                # Now shift the final output x2_0 into the inputs:
                with name_scope("store_to_input"):
                    x = tf.concat(1, [inputs[:, 1:, :] , x])
                    tf.assign(inputs, x)
            return x
