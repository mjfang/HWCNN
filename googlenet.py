import tensorflow as tf

def inception_module(input_layer, nfilters, scope):
    with tf.variable_scope(scope):
        mp1 = tf.layers.max_pooling2d(input_layer, 3, 1, padding='same')
        out_1 = tf.layers.conv2d(mp1, nfilters[0], 1, padding='same')

        out_2 = tf.layers.conv2d(input_layer, nfilters[1], 1, padding='same')
        
        conv1x1_0 = tf.layers.conv2d(input_layer, nfilters[2], 1, padding='same')
        out_3 = tf.layers.conv2d(conv1x1_0, nfilters[3], 3, padding='same')
        
        conv1x1_1 = tf.layers.conv2d(input_layer, nfilters[4], 1, padding='same')
        out_4 = tf.layers.conv2d(conv1x1_1, nfilters[5], 5, padding='same')
        
        return tf.concat([out_1, out_2, out_3, out_4], axis=3)

def googleNet(input_layer):
    with tf.variable_scope("inception"):
        out = tf.layers.conv2d(input_layer, 64, 7, strides=2, padding='same')
        out = tf.layers.max_pooling2d(out, 3, 2, padding='valid')
        out = tf.nn.local_response_normalization(out, alpha=0.00002, bias=1)
        out = tf.layers.conv2d(out, 64, 1, padding='valid')
        out = tf.layers.conv2d(out, 192, 3, padding='valid')
        out = tf.nn.local_response_normalization(out, alpha=0.00002, bias=1)
        out = tf.layers.max_pooling2d(out, 3, 2, padding='valid')
        out = inception_module(out, [32, 64, 96, 128, 16, 32], "inception1")
        out = inception_module(out, [64, 128, 128, 192, 32, 96], "inception2")
        out = tf.layers.max_pooling2d(out, 3, 2, padding='valid')
        out = inception_module(out, [64, 192, 96, 208, 16, 48], "inception3")
        out = inception_module(out, [64, 160, 112, 224, 24, 64], "inception4")
        out = inception_module(out, [64, 128, 128, 256, 24, 64], "inception5")
        out = inception_module(out, [64, 112, 144, 288, 32, 64], "inception6")
        out = inception_module(out, [128, 256, 160, 320, 32, 128], "inception7")
        out = tf.layers.max_pooling2d(out, 3, 2, padding='valid')
        out = inception_module(out, [128, 256, 160, 320, 32, 128], "inception8")
        out = inception_module(out, [128, 384, 192, 384, 48, 128], "inception9")
        out = tf.reduce_mean(out, axis=[1, 2])
        out = tf.layers.dense(out, 1000, activation=None)
        return out

