#!/usr/bin/env python
'''
A quite bogus LSTM duration model

Carl Quillen
'''

from __future__ import print_function

import optparse
import sys
import time
import math
import random
import threading
import numpy as np
from operator import mul

import tensorflow as tf
import tensorflow.contrib.layers as layers

# Options from the command line:
parser = optparse.OptionParser()
parser.add_option('-p', '--param_file', dest='param_file',
                  default=None, help='File to set parameters')
parser.add_option('-i', '--input_file', dest='input_file',
                  default=None, help='Input checkpoint file')
parser.add_option('-o', '--output_file', dest='output_file',
                  default='output_dur.txt', help='Output duration file')
parser.add_option('-l', '--lexicon', dest='lexicon',
                  default='align_lexicon.txt', help='Pronunciation lexicon')
parser.add_option('-m', '--phone_map_file', dest='phone_map_file',
                  default='phone.map', help='VCTK phone map.')
parser.add_option('-c', '--checkpoint_rate', dest='checkpoint_rate',
                  type=int, default=10000, help='Rate to checkpoint.')
parser.add_option('-s', '--summary_rate', dest='summary_rate',
                  type=int, default=20, help='Rate to output summaries.')
parser.add_option('-L', '--base_learning_rate', dest='base_learning_rate',
                  type=float, default=1e-03,
                  help='The initial learning rate. ' +
                  'lr = base_learning_rate/(1.0+lr_offet+timestep)*const)')
parser.add_option('-O', '--lr_offset', dest='lr_offset', type=int, default=0,
                  help="lr=base_learning_rate/(1.0+timestep+lr_offset)*const)")

opts, cmdline_args = parser.parse_args()

# Options that can be set in a parameter file:
opts.canonical_epoch_size = 2000.0
opts.num_units = 128
opts.chunk_size = 50
opts.n_chunks = 32              # chunks in training.
opts.clip = None                # Derivative clipping
opts.max_silence = 20
opts.silence_phone = 0

# Set opts.* parameters from a parameter file if you want:
if opts.param_file is not None:
    with open(opts.param_file) as f:
        exec(f)

sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))

phone_map = {}
with open(opts.phone_map_file) as p:
    for line in p:
        val, phone = line.rstrip().split()
        phone_map[phone] = int(val)
n_phones = len(phone_map)

lexicon = {}
with open(opts.lexicon) as l:
    for line in l:
        x = line.rstrip().split()
        word = x.pop(0)
        x.pop(0)
        phones = [(phone_map[k] if k in phone_map
                   else opts.silence_phone) for k in x]
        lexicon[word] = phones

cell = tf.contrib.rnn.LSTMCell(num_units=opts.num_units, num_proj=1)
train_state = cell.zero_state(opts.n_chunks, tf.float32)
state_list = []
for i, s in enumerate(cell.zero_state(opts.n_chunks, tf.float32)):
    train_var = tf.get_variable(initializer=s, name="cur_state_"+str(i))
    state_list.append(train_var[0:1, :])

print("Sentence to generate:", end='')
sys.stdout.flush()
words = sys.stdin.readline().rstrip().split()
plist = [opts.silence_phone]
for word in words:
    word = word.upper()
    if word not in lexicon:
        print("Lexicon lookup failed for", word)
        next
    plist.extend(lexicon[word])
plist.append(opts.silence_phone)
phones = np.array(plist, dtype=np.int32).repeat(2)[1:]
phones = np.append(phones, opts.silence_phone).reshape(-1, 2)


with tf.name_scope("input_massaging"):
    phone_vec = tf.one_hot(phones, depth=n_phones, axis=-1)
    phone_vec = tf.reshape(phone_vec, (1, -1, n_phones*2))

# Define the computational graph.
output, state = tf.nn.dynamic_rnn(
    cell=cell, initial_state=tf.contrib.rnn.LSTMStateTuple(*state_list),
    inputs=phone_vec)

# That should have created all training variables.  Now we can make a saver.
saver = tf.train.Saver(tf.trainable_variables())
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

# Initialize everything.
sess.run(init)

if opts.input_file is not None:
    print("Restoring from", opts.input_file)
    saver.restore(sess, opts.input_file)

durs = sess.run(output).reshape(-1)
sess.close()

assert len(durs) == len(plist)
for i, p in enumerate(plist):
    print("{}:{}".format(p, durs[i]), end=' ')
print()
