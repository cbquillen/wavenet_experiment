from __future__ import division

#
# wavenet routines, some borrowed from ibab's wavenet implementation
#

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.framework import arg_scope, add_arg_scope


@add_arg_scope
def causal_atrous_conv1d(*args, **kwargs):
    '''
    Make convolution causal by shifting the output right
    by the correct number of samples.  This happens in
    two stages:
        1) pad the input to the left by half the (kernel size-1).
        2) extract the right part of the enlarged output.
    '''
    # Only three arguments are allowed un-named.  The first three:
    if len(args) > 0:
        kwargs['inputs'] = args[0]
    if len(args) > 1:
        kwargs['num_outputs'] = args[1]
    if len(args) > 2:
        kwargs['kernel_size'] = args[2]

    rate = kwargs['rate']
    # From experiment, 2-point convolutions are not causal. That means
    # that even-with stencils need to be treated like the next
    # larger odd filter.  This should be correct:
    pad_amount = (kwargs['kernel_size']//2*rate)
    inputs = kwargs['inputs']

    # The inputs are a three-dimensional tensor,
    # because of the channels and output dimensions.
    assert len(inputs.get_shape()) == 3  # rank 3!

    with tf.name_scope(kwargs['scope']+'_pad'):
        inputs = tf.pad(inputs, [[0, 0], [pad_amount, 0], [0, 0]])
    kwargs['inputs'] = inputs
    out = layers.convolution(**kwargs)
    with tf.name_scope(kwargs['scope']+'_slice'):
        out = out[:, 0:-pad_amount, :]
    return out


@add_arg_scope
def generate_conv1d(*args, **kwargs):
    '''
    For generation, we only need to generate one sample.
    We are going to directly store it via tf.assign() in
    the variable output of the next layer, so we don't
    need to do any padding.  We'll also need padding=VALID
    to be set in the convolution itself.
    '''

    kwargs['padding'] = 'VALID'  # Force it!

    # Only three arguments are allowed un-named.  The first three:
    if len(args) > 0:
        kwargs['inputs'] = args[0]
    if len(args) > 1:
        kwargs['num_outputs'] = args[1]
    if len(args) > 2:
        kwargs['kernel_size'] = args[2]

    rate = kwargs['rate']
    with tf.name_scope(kwargs['scope']+'_end_slice'):
        inputs = kwargs['inputs'][-kernel_size*rate:]
        kwargs['inputs'] = inputs

    out = layers.convolution(**kwargs)
    assert out.get_shape()[1] == 1

    return out


def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = quantization_channels - 1
        # Perform mu-law companding transformation (ITU-T, 1988).
        audio = tf.clip_by_value(audio, -1.0, 1.0)
        magnitude = tf.log(1.0 + mu * tf.abs(audio)) / tf.log(1.0 + mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1.0) / 2.0 * mu + 0.5, tf.int32)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        casted = tf.cast(output, tf.float32)
        signal = 2 * (casted / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
