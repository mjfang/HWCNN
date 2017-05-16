import tensorflow as tf
import numpy as np
import time
import siamese_net
import os
from scipy import misc

batch_size = 25

input_width = 2270
input_height = 342

def get_random_exclude(low, high, exclude):
  r = np.random.randint(low, high)
  while r == exclude:
    r = np.random.randint(low, high)
  return r

#assume
def get_dataset(data_dir, writer_index_low, writer_index_high, num_examples):
  pairs = []
  labels = []
  writer_folders = os.listdir(data_dir)
  for i in range(num_examples / 2):
    #positive pair
    wr_1 = np.random.randint(writer_index_low, writer_index_high)
    url_folder = os.path.join(data_dir, writer_folders[wr_1])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx1 = np.random.randint(0, num_urls)
    file_idx2 = get_random_exclude(0, num_urls, file_idx1)

    #TODO: load image

    image1 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx1])), (input_height, input_width, 1))
    image2 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx2])), (input_height, input_width, 1))
    pairs += [[image1, image2]]

    #negative pair
    wr_1 = np.random.randint(writer_index_low, writer_index_high)
    url_folder = os.path.join(data_dir, writer_folders[wr_1])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx3 = np.random.randint(0, num_urls)
    image3 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx3])), (input_height, input_width, 1))

    wr_2 = get_random_exclude(writer_index_low, writer_index_high, wr_1)
    url_folder = os.path.join(data_dir, writer_folders[wr_2])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx4 = np.random.randint(0, num_urls)
    image4 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx4])), (input_height, input_width, 1))
    pairs += [[image3, image4]]
    labels += [1,0]
  return np.array(pairs),np.array(labels)

#x, labels = get_dataset("test_data", 0, 4, 300)
#print(x.shape, labels.shape)
#print("")
#
#
def compute_accuracy(prediction,labels):
  return labels[prediction.ravel() < 0.5].mean()

#657 writers
#use different writers for training set, validation set, test_set
def create_data(data_dir, num_train, num_val, num_test):

  ##proportion the writers in the same fraction as the number of examples
  num_writers = len(os.listdir(data_dir))
  train_frac = num_train / float(num_train + num_val + num_test)
  val_frac = num_val / float(num_train + num_val + num_test)
  test_frac = num_test / float(num_train + num_val + num_test)

  cutoff_1 = np.round(train_frac * num_writers)
  cutoff_2 = cutoff_1 + np.round(val_frac * num_writers)
  x_train, y_train = get_dataset(data_dir, 0, cutoff_1, num_train)
  x_val, y_val = get_dataset(data_dir, cutoff_1, cutoff_2, num_val)
  x_test, y_test = get_dataset(data_dir, cutoff_2, num_writers, num_test)

  return (x_train, y_train, x_val, y_val, x_test, y_test)


def get_next_batch(start, end, inputs, labels):
  input1 = inputs[start:end, 0]
  input2 = inputs[start:end, 1]
  y_batch = labels[start:end]
  return input1, input2, y_batch


start_time = time.time()
x_train, y_train, x_val, y_val, x_test, y_test = create_data("preprocessed_lines_contrast_adjustment", 500, 100, 100)
print("finished getting data", time.time() - start_time)
#data
with tf.Session() as sess:
  siamese = siamese_net.Siamese_Net(batch_size, [None, input_height, input_width, 1])
  train_step = tf.train.AdamOptimizer().minimize(siamese.loss)
  sess.run(tf.global_variables_initializer())

  for epoch in range(30):
    avg_loss = 0.
    avg_acc = 0.
    total_batch = int(x_train.shape[0]/batch_size)
    start_time = time.time()
    for i in range(total_batch):
      start = i * batch_size
      end = (i+1) * batch_size
      input1, input2, y = get_next_batch(start, end, x_train, y_train)
      _, loss_v, predict = sess.run([train_step, siamese.loss, siamese.distance], feed_dict={
        siamese.x1:input1,
        siamese.x2:input2,
        siamese.y : y
      })
      tr_acc = compute_accuracy(predict, y)

      if np.isnan(loss_v):
        print('Model diverged with NaN loss')
        quit()
      avg_loss += loss_v
      avg_acc += tr_acc * 100
      #if i % 10 == 0:
      #  ('step %d: loss %.3f' % (i, loss_v))
    duration = time.time() - start_time


    print('epoch %d  time: %f loss %0.5f acc %0.2f' %(epoch,duration,avg_loss/(total_batch),avg_acc/total_batch))

  #test
  input1, input2, y = get_next_batch(0, x_test.shape[0], x_test, y_test)
  predict = siamese.distance.eval(feed_dict={
    siamese.x1:input1,
    siamese.x2:input2,
    siamese.y: y
  })
  test_acc = compute_accuracy(predict, y)
  print("Test set accuracy %0.2f" % (100 * test_acc))

