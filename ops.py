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
    pad_amount = (kwargs['kernel_size']*rate)//2
    inputs = kwargs['inputs']

    # The inputs are a three-dimensional tensor, because of the channels and output dimensions.
    assert len(inputs.get_shape()) == 3	 # rank 3!

    inputs = tf.pad(inputs, [[0, 0], [pad_amount, 0], [0, 0]])
    out = layers.convolution(kwargs)
    return tf.slice(out, [0, 0, 0], [-1, tf.shape(inputs)[1], -1])

