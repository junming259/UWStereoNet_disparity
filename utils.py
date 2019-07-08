import tensorflow as tf


def deprocess(img):
    '''
    denormalize image from [-1,1] tp [0,1], and then to uint8
    '''
    img = (img + 1) / 2
    img = tf.image.convert_image_dtype(img, dtype=tf.uint8, saturate=True)
    return img


def lrelu(x, a=0.2, name="lrelu"):
    with tf.name_scope(name):
        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def conv3d_norm_lrelu(inputs, filters, kernel_size=3, strides=(1, 1, 1), is_train=True, withLrelu=True, padding='same', name='conv_norm_lrelu'):
    """
    Containing convolution, batch_norm and leaky_relu.
    """
    with tf.variable_scope(name):
        conv = tf.layers.conv3d(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False, name='conv3d')
        output = tf.layers.batch_normalization(conv, training=is_train, name='batch_norm')
        if withLrelu:
            output = lrelu(output, name='lrelu')
    return output


def deconv3d_norm_lrelu(inputs, filters, kernel_size=3, strides=(2, 2, 2), is_train=True, name='deconv_norm_lrelu'):
    """
    Containing convolution, batch_norm and leaky_relu.
    """
    with tf.variable_scope(name):
        conv = tf.layers.conv3d_transpose(inputs, filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name='deconv3d')
        normalized = tf.layers.batch_normalization(conv, training=is_train, name='batch_norm')
        output = lrelu(normalized, name='lrelu')
    return output


def conv_norm_lrelu(inputs, filters, kernel_size=3, strides=(1, 1), is_train=True, withLrelu=True, name='conv_norm_lrelu'):
    """
    Containing convolution, batch_norm and leaky_relu.
    kernel_size: 3
    strides: (1,1)
    """
    with tf.variable_scope(name):
        conv = tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name='conv')
        output = tf.layers.batch_normalization(conv, training=is_train, name='batch_norm')
        if withLrelu:
            output = lrelu(output, name='lrelu')
    return output


def deconv_norm_lrelu(inputs, filters, kernel_size, strides, is_train, name='conv_norm_lrelu'):
    """
    Containing deconvolution, batch_norm and leaky_relu.
    """
    with tf.variable_scope(name):
        conv = tf.layers.conv2d_transpose(inputs, filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name='deconv')
        normalized = tf.layers.batch_normalization(conv, training=is_train, name='batch_norm')
        output = lrelu(normalized, name='lrelu')
    return output


def norm_lrelu_conv(inputs, filters, kernel_size, strides, is_train, name='norm_lrelu_conv'):
    """
    Containing convolution, batch_norm and leaky_relu.
    """
    with tf.variable_scope(name):
        normalized = tf.layers.batch_normalization(inputs, training=is_train, name='batch_norm')
        lre = lrelu(normalized, name='lrelu')
        conv = tf.layers.conv2d(lre, filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name='conv')
    return conv


def identityBlock(inputs, filters, is_train, name='identityBlock'):
    with tf.variable_scope(name):
        inputs = tf.identity(inputs, name='inputs')
        conv1 = conv_norm_lrelu(inputs, filters[0], strides=(1, 1), is_train=is_train, name="conv1")
        conv2 = conv_norm_lrelu(conv1, filters[1], strides=(1, 1), is_train=is_train, name="conv2")
        conv3 = conv_norm_lrelu(conv2, filters[2], strides=(1, 1), is_train=is_train, withLrelu=False, name="conv3")
        output = tf.add(inputs, conv3)
        output = lrelu(output)
    return output


def convBlock(inputs, filters, strides, is_train, name='convBlock'):
    with tf.variable_scope(name):
        inputs = tf.identity(inputs, name='inputs')
        conv1 = conv_norm_lrelu(inputs, filters[0], strides=strides, is_train=is_train, name="conv1")
        conv2 = conv_norm_lrelu(conv1, filters[1], strides=(1, 1), is_train=is_train, name="conv2")
        conv3 = conv_norm_lrelu(conv2, filters[2], strides=(1, 1), is_train=is_train, withLrelu=False, name="conv3")
        shortcut = conv_norm_lrelu(inputs, filters[2], strides=strides, is_train=is_train, withLrelu=False, name="shortcut")
        output = tf.add(shortcut, conv3)
        output = lrelu(output)
    return output


def identityBlock3D(inputs, filters, is_train, name='identityBlock'):
    with tf.variable_scope(name):
        inputs = tf.identity(inputs, name='inputs')
        conv1 = conv3d_norm_lrelu(inputs, filters[0], strides=(1, 1, 1), is_train=is_train, name="conv1")
        conv2 = conv3d_norm_lrelu(conv1, filters[1], strides=(1, 1, 1), is_train=is_train, name="conv2")
        conv3 = conv3d_norm_lrelu(conv2, filters[2], strides=(1, 1, 1), is_train=is_train, withLrelu=False, name="conv3")
        output = tf.add(inputs, conv3)
        output = lrelu(output)
    return output


def convBlock3D(inputs, filters, strides, is_train, name='convBlock'):
    with tf.variable_scope(name):
        inputs = tf.identity(inputs, name='inputs')
        conv1 = conv3d_norm_lrelu(inputs, filters[0], strides=strides, is_train=is_train, name="conv1")
        conv2 = conv3d_norm_lrelu(conv1, filters[1], strides=(1, 1, 1), is_train=is_train, name="conv2")
        conv3 = conv3d_norm_lrelu(conv2, filters[2], strides=(1, 1, 1), is_train=is_train, withLrelu=False, name="conv3")
        shortcut = conv3d_norm_lrelu(inputs, filters[2], strides=strides, is_train=is_train, withLrelu=False, name="shortcut")
        output = tf.add(shortcut, conv3)
        output = lrelu(output)
    return output
