import tensorflow as tf
import numpy as np
import time
import siamese_net
import os
import datetime
import pickle
import h5py

log_file = "logfile"
batch_size = 64
num_train = 100
num_val = 100
num_test = 100
input_width = 2270 #original image sizes
input_height = 342

num_epochs = 20
input_height_modified = 250
input_width_modified = 1500
scale = 0.25
log = open(log_file, 'w')


data = pickle.load(open("dataset", "rb"))
y_train, x_val, y_val, x_test, y_test, test_record = data
x_train = h5py.File('x_train.h5', 'r')['x_train'][()]

def get_next_batch(start, end, inputs, labels):
  input1 = inputs[start:end, 0]
  input2 = inputs[start:end, 1]
  y_batch = labels[start:end]
  return input1, input2, y_batch

def compute_counts(distance, labels):
  tp = labels[distance.ravel() < 0.5].sum()  # predict close=positive, actually pos
  tn = (1 - labels)[distance.ravel() > 0.5].sum()  # predict far=negative, actually neg
  fn = labels[distance.ravel() > 0.5].sum()  # predict far=negative, but actually pos
  fp = (1 - labels)[distance.ravel() < 0.5].sum()

  return tp, tn, fp, fn

with tf.Session() as sess:
  siamese = siamese_net.Siamese_Net(batch_size,
                                    [None, int(input_height_modified * scale), int(input_width_modified * scale), 1])

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, "./model")


  te_tp = 0
  te_tn = 0
  te_fp = 0
  te_fn = 0

  total_test_batch = int(x_test.shape[0] / batch_size)
  for i in range(total_test_batch):
    start = i * batch_size
    end = (i + 1) * batch_size
    input1, input2, y = get_next_batch(start, end, x_test, y_test)
    predict = siamese.distance.eval(feed_dict={
      siamese.x1: input1,
      siamese.x2: input2,
      siamese.y: y
    })
    tp, tn, fp, fn = compute_counts(predict, y)
    te_tp += tp
    te_tn += tn
    te_fp += fp
    te_fn += fn



  print("Test set accuracy %0.2f" % (100 * (te_tp + te_tn) / (te_tp + te_tn + te_fp + te_fn)))



