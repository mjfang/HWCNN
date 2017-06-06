#Siamese Net architecture

import tensorflow as tf
import tensorflow.contrib.layers as layers
import resnet

class Siamese_Net:
  
  def __init__(self, batch_size, input_shape):
    self.x1 = tf.placeholder(tf.float32, input_shape) # TODO: change to true dimensions
    self.x2 = tf.placeholder(tf.float32, input_shape)
    self.keep_prob = 0.5
    self.batch_size = batch_size
    with tf.variable_scope("siamese") as scope:
      self.out1 = self.resnet(self.x1)
      scope.reuse_variables()
      self.out2 = self.resnet(self.x2)
    #L2 distance
    pow = tf.pow(tf.subtract(self.out1,self.out2), 2)
    #self.distance = tf.sqrt(tf.reduce_sum(pow,1,keep_dims=True))

    self.y = tf.placeholder(tf.float32, [None])
    self.loss = self.softmax_loss()
    self.train_op = self.add_optimizer_loss(self.loss)
  def network(self, input):
    self.regularizer = layers.l2_regularizer(scale=0.1)
    conv1 = layers.conv2d(input, num_outputs=32, kernel_size=[10, 10], stride=[1, 1])
    pool1 = tf.nn.max_pool(conv1,[1,2,2,1], strides = [1,2,2,1], padding='VALID') 
    conv2 = layers.conv2d(pool1, num_outputs=64, kernel_size=[8, 8], stride=[1,1])
    pool2 = tf.nn.max_pool(conv2,[1,2,2,1], strides = [1,2,2,1], padding='VALID') 
    conv3 = layers.conv2d(pool2, num_outputs = 64, kernel_size=[4,4], stride=[1,1])
    conv2_flattened = layers.flatten(conv2)
    fc1 = layers.fully_connected(conv2_flattened, 400, biases_initializer=tf.constant_initializer(0), weights_regularizer=self.regularizer)
    d_fc1 = tf.nn.dropout(fc1,self.keep_prob)
    out = layers.fully_connected(d_fc1, 200, weights_regularizer = self.regularizer)
    #d_out = tf.nn.dropout(out, self.keep_prob)
    return out
  def resnet(self, input):
    return resnet.inference_small(input, is_training=True, num_blocks=1)
  def softmax_loss(self):
    self.h = tf.concat([tf.square(self.out1-self.out2), self.out1, self.out2], axis=1)
    out = layers.fully_connected(self.h, 2, activation_fn = None)
    
    self.distance =tf.nn.softmax( out)[:, 0]
    y_int = tf.cast(self.y, tf.int32)
    # reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # reg_term = layers.apply_regularization(self.regularizer, reg_vars)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_int, logits=out)) #+ reg_term
    
  def contrastive_loss(self):
    a = self.y * tf.square(self.distance)
    a2 = (1-self.y) * tf.square(tf.maximum((1-self.distance), 0))
    return tf.reduce_sum(a + a2) / self.batch_size / 2
  def add_optimizer_loss(self, loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gvs = optimizer.compute_gradients(loss)
    gs, vs = zip(*gvs)
    self.grad_norm = tf.global_norm(gs)
    return optimizer.apply_gradients(gvs)
