#Siamese Net architecture

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Siamese_Net:

  def __init__(self, batch_size, input_shape, learning_rate=0.0001):
    self.data_dict = None
    self.var_dict = {}

    self.learning_rate = learning_rate
    self.x1 = tf.placeholder(tf.float32, input_shape) # TODO: change to true dimensions
    self.x2 = tf.placeholder(tf.float32, input_shape)
    self.keep_prob = 0.5
    self.batch_size = batch_size
    with tf.variable_scope("siamese") as scope:
      self.out1 = self.network(self.x1)
      scope.reuse_variables()
      self.out2 = self.network(self.x2)
    #L2 distance
    pow = tf.pow(tf.subtract(self.out1,self.out2), 2)
    #self.distance = tf.sqrt(tf.reduce_sum(pow,1,keep_dims=True))

    self.y = tf.placeholder(tf.float32, [None])
    self.loss = self.softmax_loss()
    self.train_op = self.add_optimizer_loss(self.loss)

  def network(self, input):
    self.regularizer = layers.l2_regularizer(scale=5e-4) # same as what the vgg19 paper uses
    pooling_padding = 'SAME'

    conv1_1 = layers.conv2d(input, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv1_2 = layers.conv2d(conv1_1, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    pool1 = tf.nn.avg_pool(conv1_2, [1,2,2,1], strides = [1,2,2,1], padding=pooling_padding)

    conv2_1 = layers.conv2d(pool1, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv2_2 = layers.conv2d(conv2_1, num_outputs=128, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    pool2 = tf.nn.avg_pool(conv2_2, [1,2,2,1], strides = [1,2,2,1], padding=pooling_padding)

    conv3_1 = layers.conv2d(pool2, num_outputs=256, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv3_2 = layers.conv2d(conv3_1, num_outputs=256, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv3_3 = layers.conv2d(conv3_2, num_outputs=256, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv3_4 = layers.conv2d(conv3_3, num_outputs=256, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    pool3 = tf.nn.avg_pool(conv3_2, [1,2,2,1], strides = [1,2,2,1], padding=pooling_padding)

    conv4_1 = layers.conv2d(pool3, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv4_2 = layers.conv2d(conv4_1, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv4_3 = layers.conv2d(conv4_2, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv4_4 = layers.conv2d(conv4_3, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    pool4 = tf.nn.avg_pool(conv4_2, [1,2,2,1], strides = [1,2,2,1], padding=pooling_padding)

    conv5_1 = layers.conv2d(pool4, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv5_2 = layers.conv2d(conv5_1, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv5_3 = layers.conv2d(conv5_2, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # conv5_4 = layers.conv2d(conv5_3, num_outputs=512, kernel_size=[3, 3], stride=[1, 1], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    pool5 = tf.nn.avg_pool(conv5_2, [1,2,2,1], strides = [1,2,2,1], padding=pooling_padding)

    pool5_flattened = layers.flatten(pool5)

    fc1 = layers.fully_connected(pool5_flattened, 4096, biases_initializer=tf.constant_initializer(0), weights_regularizer=self.regularizer, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    relu1 = tf.nn.relu(fc1)
    relu1 = tf.nn.dropout(relu1, self.keep_prob)

    fc2 = layers.fully_connected(relu1, 4096, biases_initializer=tf.constant_initializer(0), weights_regularizer=self.regularizer, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    relu2 = tf.nn.relu(fc2)
    relu2 = tf.nn.dropout(relu2, self.keep_prob)

    fc3 = layers.fully_connected(relu2, 1000, biases_initializer=tf.constant_initializer(0), weights_regularizer=self.regularizer, weights_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    relu3 = tf.nn.relu(fc3)
    relu3 = tf.nn.dropout(relu3, self.keep_prob)
    return relu3

  def avg_pool(self, bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
      filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, conv_biases)
      relu = tf.nn.relu(bias)

      return relu

  def fc_layer(self, bottom, in_size, out_size, name):
    with tf.variable_scope(name):
      weights, biases = self.get_fc_var(in_size, out_size, name)

      x = tf.reshape(bottom, [-1, in_size])
      fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

      return fc

  def get_conv_var(self, filter_size, in_channels, out_channels, name):
      initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
      filters = self.get_var(initial_value, name, 0, name + "_filters")

      initial_value = tf.truncated_normal([out_channels], .0, .001)
      biases = self.get_var(initial_value, name, 1, name + "_biases")

      return filters, biases

  def get_fc_var(self, in_size, out_size, name):
      initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
      weights = self.get_var(initial_value, name, 0, name + "_weights")

      initial_value = tf.truncated_normal([out_size], .0, .001)
      biases = self.get_var(initial_value, name, 1, name + "_biases")

      return weights, biases

  def get_var(self, initial_value, name, idx, var_name):
      if self.data_dict is not None and name in self.data_dict:
          value = self.data_dict[name][idx]
      else:
          value = initial_value

      var = tf.Variable(value, name=var_name)

      self.var_dict[(name, idx)] = var

      # print var_name, var.get_shape().as_list()
      assert var.get_shape() == initial_value.get_shape()

      return var

  def softmax_loss(self):
    self.h = tf.concat([tf.square(self.out1-self.out2), self.out1, self.out2], axis=1)
    out = layers.fully_connected(self.h, 2, activation_fn = None)

    self.distance =tf.nn.softmax( out)[:, 0]
    y_int = tf.cast(self.y, tf.int32)
    reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = layers.apply_regularization(self.regularizer, reg_vars)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_int, logits=out)) + reg_term

  def contrastive_loss(self):
    a = self.y * tf.square(self.distance)
    a2 = (1-self.y) * tf.square(tf.maximum((1-self.distance), 0))
    return tf.reduce_sum(a + a2) / self.batch_size / 2

  def add_optimizer_loss(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    gvs = optimizer.compute_gradients(loss)
    gs, vs = zip(*gvs)
    self.grad_norm = tf.global_norm(gs)
    return optimizer.apply_gradients(gvs)
