import tensorflow as tf
# import tensorflow.contrib as tf_contrib
import numpy as np
import tensorflow_addons as tfa


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

# weight_init = tf_contrib.layers.variance_scaling_initializer()
# weight_init = tf.contrib.layers.xavier_initializer()
weight_init = tf.initializers.GlorotUniform()
# weight_regularizer = tf_contrib.layers.l2_regularizer(5e-4)


##################################################################################
# Layer
##################################################################################

def ws_reg(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    # kernel_std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)

weight_regularizer = ws_reg

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0', reuse=False, focal_init=None, separable=False, use_ws_reg=True):
    # with tf.variable_scope(scope, reuse=reuse):
        if focal_init is not None:
            np_arr = np.zeros([kernel])
            np_arr[0] = focal_init
            x = tf.keras.layers.Conv2D(filters=channels,
                                kernel_size=kernel, kernel_initializer=weight_init,
                                bias_initializer=tf.constant_initializer(np_arr),
                                # kernel_regularizer=weight_regularizer,
                                strides=stride, use_bias=use_bias, padding=padding)(x)
        else:
            x = tf.keras.layers.Conv2D(channels,
                                kernel_size=kernel, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer,
                                # kernel_regularizer=ws_reg,
                                strides=stride, use_bias=use_bias, padding=padding)(x)
        return x



def upsample(x, size=(2, 2), scope='upsample_0', use_deconv=False, filters=32, kernel_size=5):
    # with tf.variable_scope(scope):
        if use_deconv:
            x = tf.keras.layers.Conv2DTranspose(filters,
                                                kernel_size,
                                                strides=size,
                                                padding='SAME',
                                                kernel_initializer=weight_init,
                                                kernel_regularizer=weight_regularizer,
                                                # kernel_regularizer=ws_reg,
                                                dilation_rate=1)(x)
            x = batch_norm(x, scope='bn_' + scope)
            x = relu(x)
        else:
            x = tf.keras.layers.UpSampling2D(size=size)(x)
        return x

def maxpool2d(x, size=(1, 2, 2, 1), scope='maxpool_0'):
    # with tf.variable_scope(scope):
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='SAME')(x)
        return x

def zeropad(x, padding, scope='zeropad_0'):
    # with tf.variable_scope(scope):
        x = tf.keras.layers.ZeroPadding2D(padding=padding)(x)
        return x

def crop(x, cropping, scope='crop_0'):
    # with tf.variable_scope(scope):
        x = tf.keras.layers.Cropping2D(cropping=cropping)(x)
        return x
        

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    # with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.keras.layers.Dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def dropout(x, rate=0.5, scope='dropout'):
    # with tf.variable_scope(scope):
        return tf.keras.layers.Dropout(rate)(x)

# def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
#     with tf.variable_scope(scope) :

#         x = batch_norm(x_init, is_training, scope='batch_norm_0')
#         x = relu(x)


#         if downsample :
#             x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
#             x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

#         else :
#             x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

#         x = batch_norm(x, is_training, scope='batch_norm_1')
#         x = relu(x)
#         x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')

#         return x + x_init

def resblock(x_init, channels, use_bias=False, downsample=False, scope='resblock', bev=False) :
    # with tf.variable_scope(scope) :
        stride = 2 if downsample else 1
        x = conv(x_init, channels, kernel=3, stride=stride, use_bias=use_bias, scope='conv_0')
        x = batch_norm(x, scope='batch_norm_0')
        x = relu(x)

        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')
        x = batch_norm(x, scope='batch_norm_1')

        if downsample :
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')
            x_init = batch_norm(x_init, scope='batch_norm_init')
        
        if bev :
            x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')
            x_init = batch_norm(x_init, scope='batch_norm_init')
        
        x = x + x_init
        x = relu(x)

        return x

def bottle_resblock(x_init, channels, use_bias=True, downsample=False, scope='bottle_resblock') :
    # with tf.variable_scope(scope) :
        x = batch_norm(x_init, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.keras.layers.Flatten()(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    # return leaky_relu(x)

def leaky_relu(x):
    return tf.nn.leaky_relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, scope='batch_norm', G=8):
    
    # return tf_contrib.layers.batch_norm(x,
    #                                     decay=0.9, epsilon=1e-05,
    #                                     center=True, scale=True, updates_collections=None,
    #                                     is_training=is_training, scope=scope)
    # with tf.variable_scope(scope) :
    #     return tf.layers.BatchNormalization(renorm=True)(x, training=is_training)
    return group_norm(x, G=G, eps=1e-5, scope=scope)

def group_norm(x, G=4, eps=1e-5, scope='group_norm') :
    # with tf.variable_scope(scope) :
        # N, H, W, C = x.get_shape().as_list()
        # G = min(G, C)

        # x = tf.reshape(x, [-1, H, W, G, C // G])
        # mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        # x = (x - mean) / tf.sqrt(var + eps)

        # gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        # beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))


        # x = tf.reshape(x, [-1, H, W, C]) * gamma + beta

        x = tfa.layers.GroupNormalization(groups=G, axis=3)(x)

        return x

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy



