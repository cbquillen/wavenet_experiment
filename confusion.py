import tensorflow as tf


def update_confusion(opts, prediction, reference, reuse=False):
    '''Deal with updating an exponentially-weighted-average confusion matrix
    estimate.
    '''
    dim = prediction.get_shape()[-1]
    with tf.variable_scope("confusion", reuse=reuse):
        confusion = tf.get_variable(
            'confusion',
            (dim, dim), initializer=tf.constant_initializer(),
            collections=['confusion', tf.GraphKeys.GLOBAL_VARIABLES],
            trainable=False)

        confusion_delta = tf.matmul(prediction, reference, transpose_a=True)
                                              
        confusion = tf.assign(
            confusion,
            opts.confusion_alpha*tf.cast(confusion_delta, tf.float32) +
            (1.0-opts.confusion_alpha)*confusion)

        return confusion
