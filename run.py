import tensorflow as tf
import numpy as np
import time
import siamese_net
import os
from scipy import misc
from PIL import Image

batch_size = 64

input_width = 2270
input_height = 342

def get_random_exclude(low, high, exclude):
  r = np.random.randint(low, high)
  while r == exclude:
    r = np.random.randint(low, high)
  return r

#assume
def processImage(image):
  image = image.astype(float)
  return image / 256.
#assume
def get_dataset(data_dir, writer_index_low, writer_index_high, num_examples):
  pairs = []
  labels = []
  writer_folders = os.listdir(data_dir)
  print(num_examples)
  for i in range(int(num_examples / 2)):
    #positive pair
    wr_1 = np.random.randint(writer_index_low, writer_index_high)
    url_folder = os.path.join(data_dir, writer_folders[wr_1])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx1 = np.random.randint(0, num_urls)
    file_idx2 = get_random_exclude(0, num_urls, file_idx1)

    #TODO: load image

    image1 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx1])), (input_height, input_width))
    image2 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx2])), (input_height, input_width))
    image1 = image1[:, :input_height]
    image2 = image2[:, :input_height]
    image1 = misc.imresize(image1, 0.3)
    image2 = misc.imresize(image2, 0.3)
    Image.fromarray(image1, 'L').show()
    image1 = processImage(np.expand_dims(image1, axis=2))
    image2 = processImage(np.expand_dims(image2, axis=2))

    pairs += [[image1, image2]]

    #negative pair
    wr_1 = np.random.randint(writer_index_low, writer_index_high)
    url_folder = os.path.join(data_dir, writer_folders[wr_1])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx3 = np.random.randint(0, num_urls)
    image3 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx3])), (input_height, input_width))
    image3 = image3[:, :input_height]
    image3 = misc.imresize(image3, 0.3)
    wr_2 = get_random_exclude(writer_index_low, writer_index_high, wr_1)
    url_folder = os.path.join(data_dir, writer_folders[wr_2])
    urls = os.listdir(url_folder)
    num_urls = len(urls)
    file_idx4 = np.random.randint(0, num_urls)
    image4 = np.reshape(misc.imread(os.path.join(url_folder, urls[file_idx4])), (input_height, input_width))
    image4 = image4[:, :input_height]

    image4 = misc.imresize(image4, 0.3)
    image3 = processImage(np.expand_dims(image3, axis=2))
    image4 = processImage(np.expand_dims(image4, axis=2))
    pairs += [[image3, image4]]
    labels += [1,0]
  return np.array(pairs),np.array(labels)


#x, labels = get_dataset("test_data", 0, 4, 300)
#print(x.shape, labels.shape)
#print("")
#
#
def compute_counts(prediction,labels):
  
  tp = labels[prediction.ravel() < 0.5].sum()
  tn = (1-labels)[prediction.ravel() > 0.5].sum()
  fp = labels[prediction.ravel() > 0.5].sum()
  fn = (1-labels)[prediction.ravel() < 0.5].sum()
  
  return tp, tn, fp, fn

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
x_train, y_train, x_val, y_val, x_test, y_test = create_data("preprocessed_lines_contrast_adjustment", 500, 200, 200)
print("finished getting data", time.time() - start_time)
#data

def do_summaries(siamese, graph, output_file = "results/"):
  tf.summary.scalar("loss", siamese.loss)
  tf.summary.scalar("grads norm", siamese.grad_norm)
  merged = tf.summary.merge_all()
  return merged, tf.summary.FileWriter(output_file, graph)

with tf.Session() as sess:
  siamese = siamese_net.Siamese_Net(batch_size, [None, input_height, input_height, 1])
  
  #train_step = tf.train.AdamOptimizer().minimize(siamese.loss)
  merged, fw = do_summaries(siamese, sess.graph) 
  sess.run(tf.global_variables_initializer())
  counter = 0
  for epoch in range(30):
    avg_loss = 0.
    count_tp = 0.
    count_tn = 0.
    count_fp = 0.
    count_fn = 0.
    total_batch = int(x_train.shape[0]/batch_size)
    start_time = time.time()
    for i in range(total_batch):
      start = i * batch_size
      end = (i+1) * batch_size
      input1, input2, y = get_next_batch(start, end, x_train, y_train)
      summary, _, loss_v, predict = sess.run([merged, siamese.train_op, siamese.loss, siamese.distance], feed_dict={
        siamese.x1:input1,
        siamese.x2:input2,
        siamese.y : y
      })
      fw.add_summary(summary, counter)
      tr_tp, tr_tn, tr_fp, tr_fn = compute_counts(predict, y)
      count_tp += tr_tp
      count_tn += tr_tn
      count_fp += tr_fp
      count_fn += tr_fn
      if np.isnan(loss_v):
        print('Model diverged with NaN loss')
        quit()
      avg_loss += loss_v
      #avg_acc += tr_acc * 100
      #if i % 10 == 0:
      #  ('step %d: loss %.3f' % (i, loss_v))
    duration = time.time() - start_time
    acc = (count_tp + count_tn) / (count_tp + count_tn + count_fp + count_fn)  
    print('epoch %d  time: %f loss %0.5f acc %0.2f tp %f tn %f fp %f fn %f' %(epoch,duration,avg_loss/(total_batch),acc, count_tp, count_tn, count_fp, count_fn))
    #val
    v_tp = 0.
    v_tn = 0.
    v_fp = 0.
    v_fn = 0.
    total_val_batch = int(x_val.shape[0]/batch_size)
    for i in range(total_val_batch):
      start = i * batch_size
      end = (i+1) * batch_size

      input1, input2, y = get_next_batch(start, end, x_val, y_val)
      predict_val = siamese.distance.eval(feed_dict={siamese.x1:input1, siamese.x2:input2, siamese.y: y})
      tp, tn, fp, fn= compute_counts(predict_val, y)
      v_tp += tp
      v_tn += tn
      v_fp += fp
      v_fn += fn
    print("Val set accuracy %0.2f tp %f tn %f fp %f fn %f" % (100 * (v_tp + v_tn) / (v_tp + v_tn + v_fp + v_fn), v_tp, v_tn, v_fp, v_fn))
  #test
  te_tp = 0.
  te_tn = 0.
  te_fp = 0.
  te_fn = 0.

  total_test_batch = int(x_test.shape[0]/batch_size)
  for i in range(total_test_batch):
    start = i * batch_size
    end = (i+1) * batch_size 
    input1, input2, y = get_next_batch(start, end, x_test, y_test)
    predict = siamese.distance.eval(feed_dict={
      siamese.x1:input1,
      siamese.x2:input2,
      siamese.y: y
    })
    tp, tn, fp, fn = compute_counts(predict, y)
    te_tp += tp
    te_tn += tn
    te_fp += fp
    te_fn += fn

  print("Test set accuracy %0.2f" % (100 * (te_tp + te_tn) / (te_tp + te_tn + te_fp + te_fn)))

