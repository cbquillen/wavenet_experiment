from __future__ import division

# wavenet routines, borrowed from ibab's wavenet implementation
# https://github.com/ibab/tensorflow-wavenet

import tensorflow as tf


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
        magnitude = (1.0 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
