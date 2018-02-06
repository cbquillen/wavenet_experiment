#!/usr/bin/env python
'''
This is a quick demo hack of a wavenet trainer, with some help
of infrastructure from ibab.

Carl Quillen
'''

from __future__ import print_function

import optparse
import sys
import time
from operator import mul

import tensorflow as tf
import tensorflow.contrib.layers as layers
from audio_reader import AudioReader
from wavenet import wavenet, compute_overlap

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
                  default='slt_f0b.txt', help='Corpus database file')
parser.add_option('-c', '--checkpoint_rate', dest='checkpoint_rate',
                  type=int, default=1000, help='Rate to checkpoint.')
parser.add_option('-s', '--summary_rate', dest='summary_rate',
                  type=int, default=20, help='Rate to output summaries.')
parser.add_option('-S', '--silence_threshold', dest='silence_threshold',
                  type=float, default=0.2,
                  help='Silence classifier energy threshold')
parser.add_option('-Z', '--audio_chunk_size', dest='audio_chunk_size',
                  type=int, default=500, help='Audio chunk size per batch.')
parser.add_option('-L', '--base_learning_rate', dest='base_learning_rate',
                  type=float, default=1e-03,
                  help='The initial learning rate. ' +
                  'lr = base_learning_rate/(1.0+lr_offet+timestep)*const)')
parser.add_option('-O', '--lr_offset', dest='lr_offset', type=int, default=0,
                  help="lr=base_learning_rate/(1.0+timestep+lr_offset)*const)")
parser.add_option('-H', '--histogram_summaries', dest='histogram_summaries',
                  action='store_true', default=False,
                  help='Do histogram summaries')

opts, cmdline_args = parser.parse_args()

# Options that can be set in a parameter file:
opts.canonical_epoch_size = 5000.0
opts.n_chunks = 10           # How many utterance chunks to train at once.
opts.input_kernel_size = 64  # The size of the input layer kernel.
opts.kernel_size = 4         # The size of other kernels.
opts.num_outputs = 128       # The number of convolutional channels.
opts.num_outputs2 = opts.num_outputs  # The "inner" convolutional channels.
opts.skip_dimension = 512    # The dimension for skip connections.
opts.dilations = [[2**N for N in range(8)]] * 5
opts.epsilon = 1e-4          # Adams optimizer epsilon.
opts.max_steps = 400000
opts.sample_rate = 16000
opts.max_checkpoints = 30
opts.reverse = False  # not used in this version..
opts.clip = 0.1
opts.context = 3      # 2 == biphone, 3 == triphone.
opts.n_phones = 41
opts.n_users = 1
opts.n_mfcc = 20
opts.mfcc_weight = 0.001
opts.nopad = False      # True to use training without the padding method.
opts.dropout = 0.0
opts.feature_noise = 1e-6


# Set opts.* parameters from a parameter file if you want:
if opts.param_file is not None:
    with open(opts.param_file) as f:
        exec(f)

# smaller audio chunks increase the timesteps per epoch:
# this is normalized relative to a 100000 sample chunk.
opts.canonical_epoch_size *= 100000.0/(opts.audio_chunk_size*opts.n_chunks)

sess = tf.Session()

coord = tf.train.Coordinator()  # Is this used for anything?

data = AudioReader(opts.data_list, coord, sample_rate=opts.sample_rate,
                   chunk_size=opts.audio_chunk_size,
                   overlap=0, reverse=False,
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

    # We will try to predict the batch from a slightly
    # noisier version on the input.
    orig_batch = batch
    if opts.feature_noise > 0:
        batch += tf.random_normal(tf.shape(batch), stddev=opts.feature_noise)

    batch = tf.expand_dims(batch, -1)

    wf_slice = slice(0, opts.audio_chunk_size)
    in_user = user[:, wf_slice] if opts.n_users > 1 else None

    mu, r, q, omfcc = wavenet(
        (batch[:, wf_slice, :], in_user, alignment[:, wf_slice],
         lf0[:, wf_slice]), opts, is_training=opts.base_learning_rate > 0)

with tf.name_scope("loss"):
    label_range = slice(1, 1+opts.audio_chunk_size)
    x = orig_batch[:, label_range]
    delta = x - mu
    the_exp = -r*tf.abs(delta) + q*delta
    loss = tf.reduce_mean(-tf.log(0.5*(r-q*q/r)) - the_exp)

if opts.logdir is not None:
    tf.summary.scalar(name="loss", tensor=loss)

# That should have created all training variables.  Now we can make a saver.
saver = tf.train.Saver(tf.trainable_variables() +
                       tf.get_collection('batch_norm'),
                       max_to_keep=opts.max_checkpoints)

if opts.histogram_summaries:
    tf.summary.histogram(name="wavenet", values=ms)
    layers.summaries.summarize_variables()

reg_loss = tf.constant(0.0)

with tf.name_scope("mfcc_loss"):
    mfcc_loss = tf.constant(0.0)
    if opts.mfcc_weight > 0:
        del_mfcc = mfcc - omfcc
        mfcc_loss = tf.reduce_mean(del_mfcc*del_mfcc)

        if opts.logdir is not None:
            tf.summary.scalar(name='mfcc', tensor=mfcc_loss)

with tf.name_scope("reg_loss"):
    if 'l2reg' in vars(opts):
        reg_loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

learning_rate = tf.placeholder(tf.float32, shape=())
# adams_epsilon probably should be reduced near the end of training.
adams_epsilon = tf.placeholder(tf.float32, shape=())

# We might want to run just measuring loss and not training,
# perhaps to see what the loss variance is on the training.
# in that case, set opts.base_learning_rate=0
if opts.base_learning_rate > 0:
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           epsilon=adams_epsilon)
        with tf.get_default_graph().control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            sum_loss = loss + opts.mfcc_weight*mfcc_loss + reg_loss
            if opts.clip is not None:
                gradients = optimizer.compute_gradients(
                    sum_loss, var_list=tf.trainable_variables())
                clipped_gradients = [
                    (tf.clip_by_value(var, -opts.clip, opts.clip)
                        if var is not None else None, name)
                    for var, name in gradients]
                minimize = optimizer.apply_gradients(clipped_gradients)
            else:
                minimize = optimizer.minimize(
                    sum_loss, var_list=tf.trainable_variables())
else:
    minimize = tf.constant(0)   # a noop.

if opts.logdir is not None:
    summaries = tf.summary.merge_all()

init = tf.global_variables_initializer()

# Finalize the graph, so that any new ops cannot be created.
# this is good for avoiding memory leaks.
tf.get_default_graph().finalize()

print("Model variables:")
total_params = 0
for var in tf.trainable_variables() + tf.get_collection('batch_norm'):
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

    if (global_step + 1) % opts.summary_rate == 0 and opts.logdir is not None:
        cur_loss, cur_mfcc_loss, summary_pb = sess.run(
            [loss, mfcc_loss, summaries, minimize],
            feed_dict={learning_rate: cur_lr,
                       adams_epsilon: opts.epsilon})[0:3]
        summary_writer.add_summary(summary_pb, global_step)
    else:
        cur_loss, cur_mfcc_loss = sess.run(
                            [loss, mfcc_loss, minimize],
                            feed_dict={learning_rate: cur_lr,
                                       adams_epsilon: opts.epsilon})[0:2]
    new_time = time.time()
    print(("loss[{}]: {:.3f} mfcc {:.3f} dt {:.3f} lr {:.4g}").format(
          global_step, cur_loss, cur_mfcc_loss, new_time - last_time, cur_lr))
    last_time = new_time

    if (global_step + 1) % opts.checkpoint_rate == 0 and \
            opts.output_file is not None:
        saver.save(sess, opts.output_file, global_step)

    sys.stdout.flush()

print("Training done.")
if opts.output_file is not None:
    saver.save(sess, opts.output_file)
sess.close()
