import tensorflow as tf
import scipy.misc
import numpy as np

BATCH_SIZE = 40

def weight_variable(shape, name, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var

def bias_variable(shape, name, bias_start=0.0, trainable = True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var

def conv2d(x, output_channels, name, k_h=5, k_w=5, s=2):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        w = weight_variable(shape=[k_h, k_w, x_shape[-1], output_channels], name='weights')
        b = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME') + b
        return conv

def deconv2d(x, output_shape, name, k_h=5, k_w=5, s=2):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        w = weight_variable([k_h, k_w, output_shape[-1], x_shape[-1]], name='weights')
        bias = bias_variable([output_shape[-1]], name='biases')
        deconv = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, s, s, 1], padding='SAME') + bias
        return deconv

def fully_connect(x, channels_out, name):
    shape = x.get_shape().as_list()
    channels_in = shape[1]
    with tf.variable_scope(name):
        weights = weight_variable([channels_in, channels_out], name='weights')
        biases = bias_variable([channels_out], name='biases')
        return tf.matmul(x, weights) + biases

def lrelu(x, leak=0.02):
    return tf.maximum(x, leak * x)

def conv_cond_concat(value, cond):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3)

# z:?*100, y:?*10
def generator(z, y, training=True):

    with tf.variable_scope("generator", reuse=not training):
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 4], name="yb")  # y:?*1*1*4
        z = tf.concat([z, y], 1)  # z:?*104

        h1 = fully_connect(z, 1024, name='g_h1_fully_connect')
        h1 = lrelu(tf.layers.batch_normalization(h1, training=training, name='g_h1_batch_norm'))
        h1 = tf.concat([h1,y],1) # 1028
        # h2 = fully_connect(h1, 7 * 25 * 128, name='g_h2_fully_connect')
        # h2 = lrelu(tf.layers.batch_normalization(h2, training=training, name='g_h2_batch_norm'))
        # h2 = tf.reshape(h2,(BATCH_SIZE,7, 25, 128))
        # h2 = conv_cond_concat(h2, yb) # h1: 1 * 4 * 260

        # h2 = tf.image.resize_images(h1, size=(4,7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # h2 = conv2d(h2, 256, name='g_h2_deconv2d',s=1)
        # h2 = lrelu(tf.layers.batch_normalization(h2, training=training, name='g_h2_batch_norm',))  # BATCH_SIZE*2*7*256
        # h2 = conv_cond_concat(h2, yb)  # h1: BATCH_SIZE*2*7*260
        # h3 = tf.image.resize_images(h2, size=(7,25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # h3 = conv2d(h3, 128, name='g_h3_deconv2d',s=1)
        # h3 = lrelu(tf.layers.batch_normalization(h3, training=training, name='g_h3_batch_norm',))  # BATCH_SIZE*4*13*128
        # h3 = conv_cond_concat(h3, yb)  # h1: BATCH_SIZE*4*13*132
        # h4 = tf.image.resize_images(h3, size=(7,25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # h4 = conv2d(h4, 64, name='g_h4_deconv2d',s=1)
        # h4 = lrelu(tf.layers.batch_normalization(h4, training=training, name='g_h4_batch_norm',))  # BATCH_SIZE*7*25*64
        # h4 = conv_cond_concat(h4, yb)  # h1: BATCH_SIZE*7*25*68
        # h5 = tf.image.resize_images(h2, size=(14,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # h5 = conv2d(h5, 64, name='g_h5_deconv2d',s=1)
        # h5 = lrelu(tf.layers.batch_normalization(h5, training=training, name='g_h5_batch_norm',))  # BATCH_SIZE*14*50*32
        # h5 = conv_cond_concat(h5, yb)  # h1: BATCH_SIZE*14*50*32
        #
        # h6 = tf.image.resize_images(h5, size=(28,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # h6 = conv2d(h6, 1, name='g_h6_deconv2d',s=1)
        # h6 = lrelu(tf.layers.batch_normalization(h6, training=training, name='g_h6_batch_norm',))  # BATCH_SIZE*28*100*1
        # h6 = tf.nn.tanh(h6)
        h2 = fully_connect(h1, 128*7*25, name='g_h2_fully_connect')
        h2 = lrelu(tf.layers.batch_normalization(h2, training=training, name='g_h2_batch_norm',))
        h2 = tf.reshape(h2, [BATCH_SIZE, 7, 25, 128])  # h2: ?*7*7*128
        h2 = conv_cond_concat(h2, yb)  # h2: ?*7*7*132
        h3 = deconv2d(h2, output_shape=[BATCH_SIZE, 14, 50, 128], name='g_h3_deconv2d')
        h3 = lrelu(tf.layers.batch_normalization(h3, training=training, name='g_h3_batch_norm',)) # h3: ?*14*14*128
        h3 = conv_cond_concat(h3, yb)  # h3:?*14*14*138
        h4 = deconv2d(h3, output_shape=[BATCH_SIZE, 28, 100, 1], name='g_h4_deconv2d')
        h4 = tf.nn.tanh(h4)  # h4: ?*28*100*1
        return h4

def discriminator(image, y, reuse=False, training=True):
    # with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 4], name='yb')  # BATCH_SIZE*1*1*4
    x = conv_cond_concat(image, yb)  # image: BATCH_SIZE*28*100*1 ,x: BATCH_SIZE*28*100*5

    h1 = conv2d(x, 32, name='d_h1_conv2d')
    h1 = lrelu(tf.layers.batch_normalization(h1, name='d_h1_batch_norm', training=training, reuse=reuse))  # h1: BATCH_SIZE*14*50*32
    h1 = conv_cond_concat(h1, yb)  # h1: BATCH_SIZE*14*50*15

    h2 = conv2d(h1, 64, name='d_h2_conv2d')
    h2 = lrelu(tf.layers.batch_normalization(h2, name='d_h2_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*7*25*64
    h2 = conv_cond_concat(h2, yb)  # h1: BATCH_SIZE*7*25*68

    h3 = conv2d(h2, 128, name='d_h3_conv2d')
    h3 = lrelu(tf.layers.batch_normalization(h3, name='d_h3_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*4*13*128
    h3 = tf.reshape(h3, [BATCH_SIZE, -1])
    h3 = tf.concat([h3, y], 1)   # h1: BATCH_SIZE*4*13*132

    h4 =  fully_connect(h3, 1024, name='d_h4_fully_connect')
    h4 = lrelu(tf.layers.batch_normalization(h4, training=training, name='g_h4_batch_norm',))
    h4 = tf.concat([h4, y], 1)
    # h4 = conv2d(h3, 256, name='d_h4_conv2d')
    # h4 = lrelu(tf.layers.batch_normalization(h4, name='d_h4_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*2*7*256
    # h4 = conv_cond_concat(h4, yb)  # h1: BATCH_SIZE*2*7*256
    # h5 = conv2d(h4, 256, name='d_h5_conv2d')
    # h5 = lrelu(tf.layers.batch_normalization(h5, name='d_h5_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*1*4*256
    # h5 = tf.reshape(h5, [BATCH_SIZE, -1])  # BATCH_SIZE*1024
    # h5 = tf.concat([h5, y], 1)  # BATCH_SIZE*1028

    h6 = fully_connect(h4, 1, name='d_h6_fully_connect')
    # h3 = lrelu(tf.layers.batch_normalization(h3, name='d_h3_batch_norm', training=training, reuse=reuse))  # BATCH_SIZE*1024
    # h3 = tf.concat([h3, y], 1)  # BATCH_SIZE*1034
    # h4 = fully_connect(h3, 1, name='d_h4_fully_connect')  # BATCH_SIZE*1

    return tf.nn.sigmoid(h6)
    # return h4

def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)

def save_images(images, size, path):
    # normalization
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image

    return scipy.misc.imsave(path, merge_img)
