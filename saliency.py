import tensorflow as tf
import numpy as np
import time
import siamese_net
import os
import datetime
import pickle
import h5py
import matplotlib.pyplot as plt

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
x_mean = pickle.load(open("x_mean", "rb"))
y_train, x_val, y_val, x_test, y_test, test_record = data
x_train = h5py.File('x_train.h5', 'r')['x_train'][()]
def deprocessImage(image):
  image += x_mean[0]
  image = np.squeeze(image) * 256
  return image

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

def compute_saliency_maps(X, y, model):
  saliency = None
  a = tf.stack((tf.range(X.shape[0]), tf.cast(model.y, tf.int32)), axis=1)
  correct_scores = tf.gather_nd(model.scores,
                                a)
  grads = tf.gradients(model.loss, [model.x1, model.x2])
  # grads_x2 = tf.gradients(model.loss, model.x2)
  gr_vals = sess.run(grads, feed_dict={model.x1: X[:, 0], model.x2: X[:, 1], model.y: y})
  saliency1 = np.max(np.abs(gr_vals[0]), axis=3)
  saliency2 = np.max(np.abs(gr_vals[1]), axis=3)

  return saliency1, saliency2

def show_saliency_maps(X,y,mask, model):
  mask = np.asarray(mask)
  Xm = X[mask]
  ym = y[mask]

  saliency = compute_saliency_maps(Xm, ym, model)

  for i in range(mask.size):
    plt.subplot(2, mask.size, i + 1)
    plt.imshow(deprocessImage(Xm[i][0]), cmap='gray')

    plt.axis('off')
    # plt.title(class_names[ym[i]])
    plt.subplot(2, mask.size, mask.size + i + 1)
    plt.title(mask[i])
    plt.imshow(saliency[0][i], cmap=plt.cm.hot)
    plt.axis('off')
    plt.gcf().set_size_inches(10, 4)
  plt.show()




with tf.Session() as sess:
  siamese = siamese_net.Siamese_Net(batch_size,
                                    [None, int(input_height_modified * scale), int(input_width_modified * scale), 1])

  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, "./model")
  mask = np.arange(5)
  show_saliency_maps(x_val, y_val, mask, siamese)




