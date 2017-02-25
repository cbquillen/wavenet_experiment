import tensorflow as tf


def update_confusion(opts, prediction, reference, reuse=False):
    '''Deal with updating an exponentially-weighted-average confusion matrix
    estimate.  The prediction and reference inputs should be mu-law encoded.
    '''
    with tf.variable_scope("confusion", reuse=reuse):
        confusion = tf.get_variable(
            'confusion',
            (opts.quantization_channels, opts.quantization_channels),
            initializer=tf.constant_initializer(),
            collections=['confusion', tf.GraphKeys.GLOBAL_VARIABLES],
            trainable=False)

        confusion_delta = tf.confusion_matrix(prediction, reference,
                                              num_classes=256)
        confusion = tf.assign(
            confusion,
            opts.confusion_alpha*tf.cast(confusion_delta, tf.float32) +
            (1.0-opts.confusion_alpha)*confusion)

        return confusion
