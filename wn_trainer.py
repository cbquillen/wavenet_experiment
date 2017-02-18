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
from ops import mu_law_encode, mu_law_decode
from wavenet import wavenet

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
parser.add_option('-d', '--data', dest='data_dir',
                  default='VCTK-Corpus', help='VCTK data directory')
parser.add_option('-c', '--checkpoint_rate', dest='checkpoint_rate',
                  type=int, default=1000, help='Rate to checkpoint.')
parser.add_option('-s', '--summary_rate', dest='summary_rate',
                  type=int, default=20, help='Rate to output summaries.')
parser.add_option('-S', '--silence_threshold', dest='silence_threshold',
                  type=float, default=0.2,
                  help='Silence classifier energy threshold')
parser.add_option('-Z', '--audio_chunk_size', dest='audio_chunk_size',
                  type=int, default=50000, help='Audio chunk size per batch.')
parser.add_option('-L', '--base_learning_rate', dest='base_learning_rate',
                  type=float, default=1e-03,
                  help='The initial learning rate. ' +
                  'lr = base_learning_rate/(lr_offet + timestep/const)')
parser.add_option('-O', '--lr_offset', dest='lr_offset',
                  type=float, default=1.0,
                  help="lr = base_learning_rate/(lr_offset + timestep/const)")
parser.add_option('-H', '--histogram_summaries', dest='histogram_summaries',
                  action='store_true', default=False,
                  help='Do histogram summaries')
parser.add_option('-b', '--batch_norm', dest='batch_norm',
                  action='store_true', default=False,
                  help='Do batch normalization')
parser.add_option('-w', '--which_future', dest='which_future',
                  type=int, default=1,
                  help='Which sample in the future to predict')

opts, cmdline_args = parser.parse_args()

# Options that can be set in a parameter file:
opts.canonical_epoch_size = 20000.0
opts.batch_size = 1         # How many utterances to train at once.
opts.input_kernel_size = 32  # The size of the input layer kernel.
opts.kernel_size = 2        # The size of other kernels.
opts.num_outputs = 64       # The number of convolutional channels.
opts.skip_dimension = 512   # The dimension for skip connections.
opts.dilations = [[1, 2, 4, 8, 16, 32, 128, 256, 512],
                  [1, 2, 4, 8, 16, 32, 128, 256, 512],
                  [1, 2, 4, 8, 16, 32, 128, 256, 512],
                  [1, 2, 4, 8, 16, 32, 128, 256, 512],
                  [1, 2, 4, 8, 16, 32, 128, 256, 512]]
opts.epsilon = 1e-4      # Adams optimizer epsilon.
opts.max_steps = 200000
opts.sample_rate = 16000
opts.quantization_channels = 256
opts.one_hot_input = False

# Set opts.* parameters from a parameter file if you want:
if opts.param_file is not None:
    with open(opts.param_file) as f:
        exec(f)

assert opts.which_future > 0 and opts.which_future < 20

# smaller audio chunks increase the timesteps per epoch:
# this is normalized relative to a 100000 sample chunk.
opts.canonical_epoch_size *= 100000/opts.audio_chunk_size


sess = tf.Session()

coord = tf.train.Coordinator()  # Is this used for anything?
data = AudioReader(opts.data_dir, coord, sample_rate=opts.sample_rate,
                   sample_size=opts.audio_chunk_size,
                   silence_threshold=opts.silence_threshold, queue_size=16)

data.start_threads(sess)         # start data reader threads.

# Define the computational graph.
with tf.name_scope("input_massaging"):
    batch = data.dequeue(num_elements=opts.batch_size)

    # We will try to predict the encoded_batch, which is a quantized version
    # of the input.
    encoded_batch = mu_law_encode(tf.reshape(batch, [opts.batch_size, -1]),
                                  opts.quantization_channels)
    if opts.one_hot_input:
        batch = tf.one_hot(encoded_batch, depth=opts.quantization_channels)

    # shift left to predict which_future samples into the future.
    encoded_batch = encoded_batch[:, opts.which_future:]

wavenet_out = wavenet(batch, opts)
for i in xrange(1, opts.which_future):
    if opts.one_hot_input:
        wavenet_out = wavenet(wavenet_out, opts, reuse=True)
    else:
        wavenet_out = tf.reshape(tf.arg_max(wavenet_out, dimension=2),
                                 shape=(-1, -1, 1))
        wavenet_out = mu_law_decode(wavenet_out, opts.quantization_channels)
        wavenet_out = wavenet(wavenet_out, opts, reuse=True)

# That should have created all training variables.  Now we can make a saver.
saver = tf.train.Saver(tf.trainable_variables())

if opts.histogram_summaries:
    tf.summary.histogram(name="wavenet", values=wavenet_out)
    layers.summaries.summarize_variables()

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=wavenet_out[:, 0:-opts.which_future], labels=encoded_batch))

tf.summary.scalar(name="loss", tensor=loss)

learning_rate = tf.placeholder(tf.float32, shape=())
# adams_epsilon probably should be reduced near the end of training.
adams_epsilon = tf.placeholder(tf.float32, shape=())
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   epsilon=adams_epsilon)
minimize = optimizer.minimize(loss, var_list=tf.trainable_variables())

summaries = tf.summary.merge_all()

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

for global_step in xrange(opts.max_steps):
    cur_lr = opts.base_learning_rate/(
        global_step/opts.canonical_epoch_size + opts.lr_offset)

    if (global_step + 1) % opts.summary_rate == 0 and opts.logdir is not None:
        cur_loss, summary_pb = sess.run([loss, summaries, minimize],
                                        feed_dict={learning_rate: cur_lr,
                                        adams_epsilon: opts.epsilon})[0:2]
        summary_writer.add_summary(summary_pb, global_step)
    else:
        cur_loss = sess.run([loss, minimize],
                            feed_dict={learning_rate: cur_lr,
                            adams_epsilon: opts.epsilon})[0]
    new_time = time.time()
    print("loss[{}]: {:.3f} dt {:.3f}".format(global_step, cur_loss,
                                              new_time - last_time))
    last_time = new_time

    if (global_step + 1) % opts.checkpoint_rate == 0 and \
            opts.output_file is not None:
        saver.save(sess, opts.output_file, global_step)
    sys.stdout.flush()

print("Training done.")
saver.save(sess, opts.output_file)
sess.close()
