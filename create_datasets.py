#!/usr/bin/python3

from os import mkdir;
from os.path import join, exists;
import pandas as pd;
import pickle;
import numpy as np;
import tensorflow as tf;

def create_datasets():

  data = pd.read_csv(join('data', 'Brain_Integ_X.csv'), skiprows = [0], header = None);
  labels = pd.read_csv(join('data', 'Brain_Integ_Y.csv'), skiprows = [1], header = 0);
  # 1) preprocess data
  survival = labels['Survival'];
  censored = labels['Censored'];
  data = np.float32(data.values);
  survival = survival.values;
  censored = censored.values;
  x = data[censored == 1]; # x.shape = (359, 399)
  y = survival[censored == 1]; # y.shape = (359,)
  labels = np.unique(y).tolist(); # sorted unique survival span
  y = [labels.index(i) for i in y]; # from survival span to label
  with open('labels.pkl', 'wb') as f:
    f.write(pickle.dumps(labels));
  # 2) divide data into training (252) and testing (107)
  idx = [i for i in range(len(y))];
  training_idx = np.random.choice(idx, size = 252, replace = False);
  testing_idx = np.setdiff1d(idx, training_idx);
  # 3) write to tfrecord
  if not exists('datasets'): mkdir('datasets');
  writer = tf.io.TFRecordWriter(join('datasets', 'trainset.tfrecord'));
  for idx in training_idx:
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'data': tf.train.Feature(float_list = tf.train.FloatList(value = x[idx,...])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [y[idx]]))
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();
  writer = tf.io.TFRecordWriter(join('datasets', 'testset.tfrecord'));
  for idx in testing_idx:
    testsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'data': tf.train.Feature(float_list = tf.train.FloatList(value = x[idx,...])),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = [y[idx]]))
      }
    ));
    writer.write(testsample.SerializeToString());
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly();
  create_datasets();
