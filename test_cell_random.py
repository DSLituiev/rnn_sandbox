#!/usr/bin/env python3
"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from  tensorflow import nn
import math
import logging
import numpy as np
from tqdm import tqdm

from six.moves import xrange  # pylint: disable=redefined-builtin

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell

if __name__ == "__main__":
    DEBUG = True
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[ %(levelname)-2s\t:%(funcName)s]\t%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if DEBUG:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    "define shapes and generate data"
    vocab_size = 8
    emb_size = 120
    num_steps = 17
    batch_size =  20
    logging.debug("batch_size\t%s" % batch_size )
    logging.debug( "num_steps\t%s" % num_steps )
    #state = tf.convert_to_tensor([3.14, 0.01])
    input_data = np.random.randint(0, vocab_size, (batch_size, num_steps))

    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [vocab_size, emb_size])

    inputs = tf.nn.embedding_lookup(embedding, input_data)

    logging.debug("embedding\t%s" % repr( embedding.get_shape() ) )
    logging.debug("inputs\t%s" % repr( inputs.get_shape() ) )

    "run test example: plug your cell class below"
    cell = BasicLSTMCell(3)

    init_state = cell.zero_state(batch_size, tf.float32)
    logging.debug("init_state\t%s" % repr( init_state.get_shape() ) )
    state = init_state
    outputs = []
    with tf.variable_scope("RNN", reuse = None):
      for time_step in tqdm(range(num_steps)):
        #print("time_step", time_step)
        if time_step > 0:
          tf.get_variable_scope().reuse_variables()
        input_slice = inputs[:, time_step,  :]
        (cell_output, state) = cell(input_slice, state)
        outputs.append(cell_output)

    print("done", outputs)

