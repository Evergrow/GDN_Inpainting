import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.contrib.layers.xavier_initializer()
weight_regularizer = None


def conv(x, channels, kernel=4, stride=2, rate=1, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, dilation_rate=rate, use_bias=use_bias)

        return x


def deconv(x, channels, kernel=4, stride=2, use_bias=True, scope='deconv'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                       kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                       strides=stride, padding='SAME', use_bias=use_bias)

        return x


def flatten(x):
    return tf.layers.flatten(x)


def max_pooling(x):
    return tf.layers.max_pooling2d(x, pool_size=2, strides=2)

##################################################################################
# Residual-block
##################################################################################


def resblock(x_init, channels, rate=1, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, rate=rate, pad=rate, pad_type='reflect', use_bias=use_bias, scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias, scope='conv')
            x = instance_norm(x, scope='ins_norm')

        return x + x_init


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def batch_norm(x, scope='batch_norm', is_train=True):
    return tf_contrib.layers.batch_norm(x,
                                        epsilon=1e-05,
                                        center=True, scale=True,
                                        is_training=is_train,
                                        scope=scope)
