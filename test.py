#!/usr/bin/python3

import tensorflow as tf;
from models import PropHazardsModel, LogLikelihood;
from create_datasets import parse_function;

batch_size = 10;

def test():

  phm = tf.keras.models.load_model('PropHazardsModel.h5', compile = False);
  ll = LogLikelihood(314);
  testset = tf.data.TFRecordDataset(join('datasets', 'testset.tfrecord')).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  for data, labels in testset:
    pred = phm(data); # pred.shape = (batch, class_num)
    loglikelihoods = ll([pred, labels]); # loglikelihoods.shape = (batch)
    loss = -tf.math.reduce_mean(loglikelihoods);
    test_loss.update_state(loss);
  print('average test loss is %f', test_loss.result());

if __name__ == "__main__":

  assert tf.executing_eagerly();
  test();
