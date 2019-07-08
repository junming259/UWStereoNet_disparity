import tensorflow as tf


class Res_bone:
    def __init__(self, imgs, is_train, reuse):
        self.imgs = imgs
        self.build(is_train, reuse)

    def build(self, is_train, reuse):
        """
        Build structure of Resnet
        """
        with tf.variable_scope('Res_bone', reuse=reuse):

            self.conv1 = self.conv_norm_lrelu(self.imgs, 32, kernel_size=7, strides=(2, 2), is_train=is_train, withLrelu=False, name="conv1")

            self.l2a = self.convBlock(self.conv1, [32, 32, 64], strides=(1, 1), is_train=is_train, name='l2a')
            self.l2b = self.identityBlock(self.l2a, [32, 32, 64], is_train=is_train, name='l2b')
            self.l2c = self.identityBlock(self.l2b, [32, 32, 64], is_train=is_train, name='l2c')

            self.l3a = self.convBlock(self.l2c, [64, 64, 64], strides=(2, 2), is_train=is_train, name='l3a')
            self.l3b = self.identityBlock(self.l3a, [64, 64, 64], is_train=is_train, name='l3b')
            self.l3c = self.identityBlock(self.l3b, [64, 64, 64], is_train=is_train, name='l3c')
            self.l3d = self.identityBlock(self.l3c, [64, 64, 64], is_train=is_train, name='l3d')

            self.l4a = self.convBlock(self.l3d, [64, 64, 128], strides=(2, 2), is_train=is_train, name='l4a')
            self.l4b = self.identityBlock(self.l4a, [64, 64, 128], is_train=is_train, name='l4b')
            self.l4c = self.identityBlock(self.l4b, [64, 64, 128], is_train=is_train, name='l4c')

            self.disp_feature = self.l3d
            self.seg_embedding = self.context_block(self.l4c, is_train=is_train)

    def context_block(self, inputs, is_train, name='context_block'):
        with tf.variable_scope(name):
            inputs = tf.identity(inputs, name='inputs')
            scale_1 = tf.layers.average_pooling2d(inputs, 2, strides=2, name='pool1')
            scale_1 = self.conv_norm_lrelu(scale_1, 32, kernel_size=1, name='scale_1')
            scale_1 = tf.image.resize_images(scale_1, tf.shape(inputs)[1:3])

            scale_2 = tf.layers.average_pooling2d(inputs, 4, strides=4, name='pool2')
            scale_2 = self.conv_norm_lrelu(scale_2, 32, kernel_size=1, name='scale_2')
            scale_2 = tf.image.resize_images(scale_2, tf.shape(inputs)[1:3])

            scale_3 = tf.layers.average_pooling2d(inputs, 8, strides=8, name='pool3')
            scale_3 = self.conv_norm_lrelu(scale_3, 32, kernel_size=1, name='scale_3')
            scale_3 = tf.image.resize_images(scale_3, tf.shape(inputs)[1:3])

            concat = tf.concat([inputs, scale_1, scale_2, scale_3], axis=-1)
            output = tf.layers.conv2d(concat, 128, 1, padding='same', use_bias=True, name='output')
        return output

    def identityBlock(self, inputs, filters, is_train, name='identityBlock'):
        with tf.variable_scope(name):
            inputs = tf.identity(inputs, name='inputs')
            conv1 = self.conv_norm_lrelu(inputs, filters[0], strides=(1, 1), is_train=is_train, name="conv1")
            conv2 = self.conv_norm_lrelu(conv1, filters[1], strides=(1, 1), is_train=is_train, name="conv2")
            conv3 = self.conv_norm_lrelu(conv2, filters[2], strides=(1, 1), is_train=is_train, withLrelu=False, name="conv3")
            output = tf.add(inputs, conv3)
            output = self.lrelu(output)
        return output

    def convBlock(self, inputs, filters, strides, is_train, name='convBlock'):
        with tf.variable_scope(name):
            inputs = tf.identity(inputs, name='inputs')
            conv1 = self.conv_norm_lrelu(inputs, filters[0], strides=strides, is_train=is_train, name="conv1")
            conv2 = self.conv_norm_lrelu(conv1, filters[1], strides=(1, 1), is_train=is_train, name="conv2")
            conv3 = self.conv_norm_lrelu(conv2, filters[2], strides=(1, 1), is_train=is_train, withLrelu=False, name="conv3")
            shortcut = self.conv_norm_lrelu(inputs, filters[2], strides=strides, is_train=is_train, withLrelu=False, name="shortcut")
            output = tf.add(shortcut, conv3)
            output = self.lrelu(output)
        return output

    def conv_norm_lrelu(self, inputs, filters, kernel_size=3, strides=(1, 1), is_train=True, withLrelu=True, name='conv_norm_lrelu'):
        """
        Containing convolution, batch_norm and leaky_relu.
        kernel_size: 3
        strides: (1,1)
        """
        with tf.variable_scope(name):
            conv = tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False, name='conv')
            output = tf.layers.batch_normalization(conv, training=is_train, name='batch_norm')
            if withLrelu:
                output = self.lrelu(output, name='lrelu')
        return output

    def lrelu(self, x, a=0.2, name="lrelu"):
        with tf.name_scope(name):
            x = tf.identity(x)
            x = (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)
        return x

    def max_pool(self, inputs, name='max_pool'):
        return tf.nn.max_pool(inputs, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
