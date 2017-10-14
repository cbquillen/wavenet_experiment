#!/usr/bin/env python
'''
Accumulate per-speaker errors.

Carl Quillen
'''

from __future__ import print_function

import optparse
import sys
import time
import numpy as np
from operator import mul

import tensorflow as tf
import tensorflow.contrib.layers as layers
from audio_reader import AudioReader
from ops import mu_law_encode, mu_law_decode
from wavenet import wavenet, wavenet_unpadded, compute_overlap

# Options from the command line:
parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-l', '--logdir', dest='logdir',
                  default=None, help='Tensorflow event logdir')
parser.add_option('-i', '--input_file', dest='input_file',
                  default=None, help='Input checkpoint file')
parser.add_option('-o', '--output_file', dest='output_file',
                  default='ckpt', help='Output checkpoint file')
parser.add_option('-d', '--data', dest='data_list',
                  default='align3.txt', help='VCTK corpus list file')
parser.add_option('-c', '--checkpoint_rate', dest='checkpoint_rate',
                  type=int, default=1000, help='Rate to checkpoint.')
parser.add_option('-s', '--summary_rate', dest='summary_rate',
                  type=int, default=20, help='Rate to output summaries.')
parser.add_option('-S', '--silence_threshold', dest='silence_threshold',
                  type=float, default=0.2,
                  help='Silence classifier energy threshold')
parser.add_option('-Z', '--audio_chunk_size', dest='audio_chunk_size',
                  type=int, default=10000, help='Audio chunk size per batch.')
parser.add_option('-L', '--base_learning_rate', dest='base_learning_rate',
                  type=float, default=1e-03,
                  help='The initial learning rate. ' +
                  'lr = base_learning_rate/(1.0+lr_offet+timestep)*const)')
parser.add_option('-O', '--lr_offset', dest='lr_offset', type=int, default=0,
                  help="lr=base_learning_rate/(1.0+timestep+lr_offset)*const)")
parser.add_option('-H', '--histogram_summaries', dest='histogram_summaries',
                  action='store_true', default=False,
                  help='Do histogram summaries')
parser.add_option('-b', '--batch_norm', dest='batch_norm',
                  action='store_true', default=False,
                  help='Do batch normalization')

opts, cmdline_args = parser.parse_args()

# Options that can be set in a parameter file:
opts.canonical_epoch_size = 20000.0
opts.n_chunks = 5           # How many utterance chunks to train at once.
opts.input_kernel_size = 32  # The size of the input layer kernel.
opts.kernel_size = 4        # The size of other kernels.
opts.num_outputs = 64       # The number of convolutional channels.
opts.num_outputs2 = opts.num_outputs  # The "inner" convolutional channels.
opts.skip_dimension = 256   # The dimension for skip connections.
opts.dilations = [[2**N for N in range(10)]] * 5
opts.epsilon = 1e-4      # Adams optimizer epsilon.
opts.max_steps = 200000
opts.sample_rate = 16000
opts.quantization_channels = 256
opts.one_hot_input = False
opts.max_checkpoints = 30
opts.clip = None
opts.which_future = 1  # Iterate prediction this many times.
opts.reverse = False  # not used in this version..
opts.user_dim = 10    # User vector dimension to use.
opts.n_phones = 183
opts.n_users = 98
opts.n_mfcc = 12
opts.mfcc_weight = 0.001
opts.nopad = False      # True to use training without the padding method.
opts.context = 3        # Triphone context.  Use 2 for biphone.

# Set opts.* parameters from a parameter file if you want:
if opts.param_file is not None:
    with open(opts.param_file) as f:
        exec(f)

# smaller audio chunks increase the timesteps per epoch:
# this is normalized relative to a 100000 sample chunk.
opts.canonical_epoch_size *= 100000.0/(opts.audio_chunk_size*opts.n_chunks)

sess = tf.Session()

coord = tf.train.Coordinator()  # Is this used for anything?

if opts.nopad:
    overlap = compute_overlap(opts)
    wavenet = wavenet_unpadded
else:
    overlap = 0

data = AudioReader(opts.data_list, coord, sample_rate=opts.sample_rate,
                   chunk_size=opts.audio_chunk_size,
                   overlap=overlap, reverse=False,
                   silence_threshold=opts.silence_threshold,
                   n_chunks=opts.n_chunks, queue_size=opts.n_chunks,
                   n_mfcc=opts.n_mfcc, context=opts.context)
assert opts.n_phones == data.n_phones
assert opts.n_users == data.n_users

data.start_threads(sess)         # start data reader threads.

# Define the computational graph.
with tf.name_scope("input_massaging"):
    batch, user, alignment, lf0, mfcc = \
        data.dequeue(num_elements=opts.n_chunks)

    if opts.n_users <= 1:
        user = None

    # We will try to predict the encoded_batch, which is a quantized version
    # of the input.
    encoded_batch = mu_law_encode(batch, opts.quantization_channels)
    if opts.one_hot_input:
        batch = tf.one_hot(encoded_batch, depth=opts.quantization_channels)
    else:
        batch = tf.reshape(batch, (opts.n_chunks, -1, 1))

wavenet_out, omfcc = wavenet((batch, user, alignment, lf0), opts,
                             is_training=opts.base_learning_rate > 0)
future_outs = [wavenet_out]
future_out = wavenet_out

for i in xrange(1, opts.which_future):
    with tf.name_scope('future_'+str(i)):
        if opts.one_hot_input:
            next_input = tf.nn.softmax(future_out)
        else:
            # Should really sample instead of arg_max...
            next_input = tf.reshape(tf.arg_max(future_out, dimension=2),
                                    shape=(opts.n_chunks, -1, 1))
            next_input = mu_law_decode(next_input, opts.quantization_channels)
        future_out, _ = wavenet_unpadded(
            (next_input, user, alignment, lf0), opts, reuse=True,
            pad_reuse=False, extra_pad_scope=str(i),
            is_training=opts.base_learning_rate > 0)
    future_outs.append(future_out)

# That should have created all training variables.  Now we can make a saver.
saver = tf.train.Saver(tf.trainable_variables(),
                       max_to_keep=opts.max_checkpoints)

if opts.histogram_summaries:
    tf.summary.histogram(name="wavenet", values=wavenet_out)
    layers.summaries.summarize_variables()

loss = reg_loss = 0
with tf.name_scope("loss"):
    uloss = tf.get_variable(
        'user_loss', initializer=tf.zeros((opts.n_users,)))
    usum = tf.get_variable(
        'user_sum', initializer=tf.zeros((opts.n_users,))+1.0)
    for i_future, future_out in enumerate(future_outs):
        if not opts.reverse:
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=future_out[:, :-(i_future+1), :],
                    labels=encoded_batch[:, overlap+i_future+1:])
            loss += tf.reduce_mean(xent)
            g_user = tf.one_hot(user[:, overlap+i_future+1:],
                                depth=opts.n_users)
            g_xent = tf.reduce_sum(
                tf.reshape(xent, (opts.n_chunks, -1, 1))*g_user, axis=[0, 1])
            uloss = tf.assign_add(uloss, g_xent)/ \
                tf.assign_add(usum, tf.reduce_sum(g_user, axis=[0, 1]))
        else:
            loss += tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=future_out[:, i_future+1:, :],
                    labels=encoded_batch[:, :-(overlap+i_future+1)]))

    loss /= opts.which_future

tf.summary.scalar(name="loss", tensor=loss)

with tf.name_scope("mfcc_loss"):
    mfcc_loss = tf.constant(0.0)
    if opts.mfcc_weight > 0:
        del_mfcc = mfcc[:, overlap:, :]-omfcc
        mfcc_loss = tf.reduce_mean(del_mfcc*del_mfcc)

        tf.summary.scalar(name='mfcc', tensor=mfcc_loss)

with tf.name_scope("reg_loss"):
    if 'l2reg' in vars(opts):
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

learning_rate = tf.placeholder(tf.float32, shape=())
# adams_epsilon probably should be reduced near the end of training.
adams_epsilon = tf.placeholder(tf.float32, shape=())

init = tf.global_variables_initializer()

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

print("Model variables:")
total_params = 0
for var in tf.trainable_variables():
    vshape = var.get_shape().as_list()
    total_params += reduce(mul, vshape)
    print("  ", var.name, vshape)
print("Total model parameters:", total_params)
sys.stdout.flush()

if opts.logdir is not None:
    summary_writer = tf.summary.FileWriter(logdir=opts.logdir,
                                           graph=tf.get_default_graph())

# Initialize everything.
sess.run(init)

if opts.input_file is not None:
    print("Restoring from", opts.input_file)
    saver.restore(sess, opts.input_file)

# Main training loop:
last_time = time.time()

for global_step in xrange(opts.lr_offset, opts.max_steps):

    # Decrease time-step by a factor of 10 for every 5 canonical epochs:
    cur_lr = opts.base_learning_rate*10.0**(
             -global_step/opts.canonical_epoch_size/5.0)

    cur_loss, cur_mfcc_loss, cur_uloss = sess.run(
                        [loss, mfcc_loss, uloss],
                        feed_dict={learning_rate: cur_lr,
                                   adams_epsilon: opts.epsilon})
    new_time = time.time()
    print("loss[{}]: {:.3f} mfcc {:.3f} dt {:.3f} lr {:.4g}".format(
        global_step, cur_loss, cur_mfcc_loss, new_time - last_time, cur_lr))
    last_time = new_time

    if (global_step + 1) % opts.checkpoint_rate == 0 and \
            opts.output_file is not None:
        saver.save(sess, opts.output_file, global_step)

    sys.stdout.flush()
    #print(cur_uloss)

print(cur_uloss)

sess.close()
