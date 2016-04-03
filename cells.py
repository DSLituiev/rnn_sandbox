"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from  tensorflow import nn
RNNCell = nn.rnn_cell.RNNCell
linear =  nn.rnn_cell.linear

import math
import logging


from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh




class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: int, The dimensionality of the inputs into the LSTM cell,
        by default equal to num_units.
    """
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias

  @property
  def input_size(self):
    return self._input_size

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return 2 * self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      c, h = array_ops.split(1, 2, state)
      concat = linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = c * sigmoid(f + self._forget_bias) + sigmoid(i) * tanh(j)
      new_h = tanh(new_c) * sigmoid(o)

    return new_h, array_ops.concat(1, [new_c, new_h])

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

    from tensorflow.models.rnn.ptb import reader
    import numpy as np

    DATAURL = "https://github.com/mil-tokyo/neural_network/tree/master/saito/sample/simple-examples/data"
    data_path = "../mldatasets/simple-examples/data/"
    try:
        raw_data = reader.ptb_raw_data( data_path)
    except:
        logging.warn("download data from\t%s\tto\t%s" % (DATAURL , data_path) )

    train_data, valid_data, test_data, _ = raw_data

    cell = BasicLSTMCell(3 )

    vocab_size = 8
    size = 120
    num_steps = 17
    batch_size =  20 
    logging.debug("batch_size\t%s" % batch_size )
    logging.debug( "num_steps\t%s" % num_steps )
    #state = tf.convert_to_tensor([3.14, 0.01])
    input_data = 1 * (np.random.rand( batch_size, num_steps, size ) < 0.05)

    with tf.device("/cpu:0"):
      embedding = tf.get_variable("embedding", [vocab_size, size])
      inputs = tf.nn.embedding_lookup(embedding, input_data) 

    logging.debug("embedding\t%s" % repr( embedding.get_shape() ) )
    logging.debug("inputs\t%s" % repr( inputs.get_shape() ) )

    init_state = cell.zero_state(batch_size, tf.float32)
    logging.debug("init_state\t%s" % repr( init_state.get_shape() ) )
    state = init_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: 
          tf.get_variable_scope().reuse_variables()
        input_slice = inputs[:,:, time_step, :]
        (cell_output, state) = cell(input_slice, state)
        outputs.append(cell_output)


    out = cell( inputs, init_state )
    print("done", out)

