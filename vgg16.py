"""Vgg16 tensorflow network.

You need to download vgg16.npy from this site
https://github.com/machrisaa/tensorflow-vgg
"""
import os
import inspect
import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:

    """Supply batch or placeholder of images [None, 224, 224, 3]
    images: scaled between 0:255
    """
    def __init__(self, images, weights=None):
        self.load_weights(weights)
        images = self.preprocess(images)
        self.pool5 = self.layers(images)
        self.fc = self.fc_layers(self.pool5)
        self.embedding = self.embedding_layer(self.fc6)
        self.probs = tf.nn.softmax(self.fc)

    def preprocess(self, images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3,
                                    value=images)
        return tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

    def layers(self, images):
        self.conv1_1 = self.conv_layer(images, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')
        return self.pool5

    def fc_layers(self, input):
        self.fc6 = self.fc_layer(input, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        return self.fc8

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def embedding_layer(self, input):
        shape = input.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        return tf.reshape(input, [-1, dim])

    def get_conv_filter(self, name):
        return tf.constant(self.weights[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.weights[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.weights[name][0], name="weights")

    def load_weights(self, weights):
        if weights is None:
            weights = inspect.getfile(Vgg16)
            weights = os.path.abspath(os.path.join(weights, os.pardir))
            weights = os.path.join(weights, "vgg16.npy")
        self.weights = np.load(weights, encoding='latin1').item()



if __name__ == "__main__":
    input = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
    vgg = Vgg16(input)
    print("conv1_1", vgg.conv1_1.shape)
    print("embedding", vgg.embedding.shape)
    print("fc6", vgg.fc6.shape)
    print("fc7", vgg.fc7.shape)
    print("fc8", vgg.fc8.shape)
    print("probs", vgg.probs.shape)
