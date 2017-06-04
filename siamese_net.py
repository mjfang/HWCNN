#Siamese Net architecture

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Siamese_Net:

  def __init__(self, batch_size, input_shape):
    self.data_dict = None
    self.var_dict = {}

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
    self.regularizer = layers.l2_regularizer(scale=0.1)
    # conv1 = layers.conv2d(input, num_outputs=32, kernel_size=[10, 10], stride=[1, 1])
    # pool1 = tf.nn.max_pool(conv1,[1,2,2,1], strides = [1,2,2,1], padding='VALID')
    # conv2 = layers.conv2d(pool1, num_outputs=64, kernel_size=[8, 8], stride=[1,1])
    # pool2 = tf.nn.max_pool(conv2,[1,2,2,1], strides = [1,2,2,1], padding='VALID')
    # conv3 = layers.conv2d(pool2, num_outputs = 64, kernel_size=[4,4], stride=[1,1])
    # conv2_flattened = layers.flatten(conv2)
    # fc1 = layers.fully_connected(conv2_flattened, 400, biases_initializer=tf.constant_initializer(0), weights_regularizer=self.regularizer)
    # d_fc1 = tf.nn.dropout(fc1,self.keep_prob)
    # out = layers.fully_connected(d_fc1, 200, weights_regularizer = self.regularizer)

    #d_out = tf.nn.dropout(out, self.keep_prob)

    self.conv1_1 = self.conv_layer(input, 1, 64, "conv1_1") # should only be one channel for b/w?
    self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
    self.pool1 = self.max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
    self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
    self.pool2 = self.max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
    self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
    self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
    self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
    self.pool3 = self.max_pool(self.conv3_4, 'pool3')

    self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
    self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
    self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
    self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
    self.pool4 = self.max_pool(self.conv4_4, 'pool4')

    self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
    self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
    self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
    self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
    self.pool5 = self.max_pool(self.conv5_4, 'pool5')

    self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
    self.relu6 = tf.nn.relu(self.fc6)
    self.relu6 = tf.nn.dropout(self.relu6, self.keep_prob)

    self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
    self.relu7 = tf.nn.relu(self.fc7)
    self.relu7 = tf.nn.dropout(self.relu7, self.keep_prob)

    self.fc8 = self.fc_layer(self.relu7, 4096, 1000, "fc8")
    self.data_dict = None

    return self.fc8

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

  def add_optimizer_loss(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    gvs = optimizer.compute_gradients(loss)
    gs, vs = zip(*gvs)
    self.grad_norm = tf.global_norm(gs)
    return optimizer.apply_gradients(gvs)
