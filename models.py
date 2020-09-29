#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;

def PropHazardsModel(dim_num, class_num):

  inputs = tf.keras.Input((dim_num,)); # inputs.shape = (batch, dim_num)
  results = tf.keras.layers.Dense(units = class_num, use_bias = True)(inputs); # results.shape = (batch, class_num)
  results = tf.keras.layers.Lambda(lambda x: tf.math.exp(x))(results); # results.shape = (batch, class_num)
  def cumsum(i, _in, _out):
    s = tf.math.reduce_sum(_in[..., i:], axis = -1, keepdims = True); # sub.shape = (batch, 1)
    _out = tf.concat([_out, s], axis = -1); # _out.shape = (batch, i)
    i += 1;
    return i, _in, _out;
  s = tf.keras.layers.Lambda(lambda x: tf.while_loop(lambda i, _in, _out: i < _in.shape[-1], cumsum, loop_vars = [0, x, tf.zeros((tf.shape(x)[0], 0), dtype = tf.float32)], shape_invariants = [tf.TensorShape([]), x.shape, tf.TensorShape([x.shape[0], None])])[2])(results); # s.shape = (batch, class_num)
  results = tf.keras.layers.Lambda(lambda x: x[0] / x[1])([results, s]); # results.shape = (batch, class_num)
  return tf.keras.Model(inputs = inputs, outputs = results);

def LogLikelihood(class_num):

  inputs = tf.keras.Input((class_num)); # inputs.shape = (batch, class_num)
  labels = tf.keras.Input((), dtype = tf.int64); # labels.shape = (batch)
  results = tf.keras.layers.Lambda(lambda x: tf.math.log(x + 1e-50))(inputs); # results.shape = (batch, class_num)
  onehot = tf.keras.layers.Lambda(lambda x, c: tf.one_hot(x, c, axis = -1), arguments = {'c': class_num})(labels); # labels.shape = (batch, class_num)
  results = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = -1))([results, onehot]); # results.shape = (batch)
  return tf.keras.Model(inputs = (inputs, labels), outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  phm = PropHazardsModel(10, 20);
  a = tf.constant(np.random.normal(size = (8, 10)));
  b = phm(a);
  print(b);
  phm.save('phm.h5');
  l = tf.constant(np.random.randint(20, size = (8)));
  ll = LogLikelihood(20);
  c = ll([b,l]);
  print(c);
  ll.save('ll.h5');
