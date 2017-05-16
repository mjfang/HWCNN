#Siamese Net architecture

import tensorflow as tf
import tensorflow.contrib.layers as layers


class Siamese_Net:

  def __init__(self, batch_size, input_shape):
    self.x1 = tf.placeholder(tf.float32, input_shape) # TODO: change to true dimensions
    self.x2 = tf.placeholder(tf.float32, input_shape)
    self.batch_size = batch_size
    with tf.variable_scope("siamese") as scope:
      self.out1 = self.network(self.x1)
      scope.reuse_variables()
      self.out2 = self.network(self.x2)
    #L2 distance
    pow = tf.pow(tf.subtract(self.out1,self.out2), 2)
    self.distance = tf.sqrt(tf.reduce_sum(pow,1,keep_dims=True))

    self.y = tf.placeholder(tf.float32, [None])
    self.loss = self.contrastive_loss()

  def network(self, input):
    conv1 = layers.conv2d(input, num_outputs=3, kernel_size=[3, 50], stride=[1, 10])
    conv2 = layers.conv2d(conv1, num_outputs=3, kernel_size=[3, 50], stride=[1,10])
    conv2_flattened = layers.flatten(conv2)
    fc1 = layers.fully_connected(conv2_flattened, 10, biases_initializer=tf.constant_initializer(0))
    out = layers.fully_connected(fc1, 2, activation_fn=None)
    return out

  def contrastive_loss(self):
    a = self.y * tf.square(self.distance)
    a2 = (1-self.y) * tf.square(tf.maximum((1-self.distance), 0))
    return tf.reduce_sum(a + a2) / self.batch_size / 2

