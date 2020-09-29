#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import tensorflow as tf;
import tensorflow_probability as tfp;
from models import PropHazardsModel, LogLikelihood;
from create_datasets import parse_function;

batch_size = 10;

def main():

  phm = PropHazardsModel(399, 314);
  ll = LogLikelihood(314);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
  # load dataset
  trainset = iter(tf.data.TFRecordDataset(join('datasets', 'trainset.tfrecord')).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  testset = iter(tf.data.TFRecordDataset(join('datasets', 'testset.tfrecord')).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
  # checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = phm, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # log
  log = tf.summary.create_file_writer('checkpoints');
  # train
  avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
  while True:
    data, labels = next(trainset);
    with tf.GradientTape() as tape:
      pred = phm(data); # pred.shape = (batch, class_num)
      loglikelihoods = ll(pred, labels); # loglikelihoods.shape = (batch)
      loss = -tf.math.reduce_mean(loglikelihoods);
    avg_loss.update_state(loss);
    if tf.equal(optimizer.iterations % 10, 0):
      with log.as_default():
        tf.summary.scalar('loss', avg_loss.result(), step = optimizer.iterations);
      print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
      if avg_loss.result() < 0.01: break;
      avg_loss.reset_states();
    grads = tape.gradient(loss, phm.trainable_variables);
    optimizer.apply_gradients(zip(grads, phm.trainable_variables));
    if tf.equal(optimizer.iterations % 10, 0):
      checkpoint.save(join('checkpoints', 'ckpt'));
      test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
      for i in range(10):
        data, labels = next(testset);
        pred = phm(data); # pred.shape = (batch, class_num)
        loglikelihoods = ll(pred, labels); # loglikelihoods.shape = (batch)
        loss = -tf.math.reduce_mean(loglikelihoods);
        test_loss.update_state(loss);
      with log.as_default():
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
  phm.save('PropHazardsModel.h5');
    
if __name__ == "__main__":

  main();
