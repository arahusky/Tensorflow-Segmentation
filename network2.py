import math
import os
import time
from math import ceil

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from imgaug import augmenters as iaa
from imgaug import imgaug
from libs.activations import lrelu
from libs.utils import corrupt

def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2] * shape[3]),
                   argmax % (shape[2] * shape[3]) // shape[3]]
    return tf.pack(output_list)


def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

class Network2:
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    IMAGE_CHANNELS = 1

    shapes = []
    encoder = []
    pooling_indices = []

    def __init__(self,
                 n_filters=[1, 64, 128, 128, 256],
                 filter_sizes=[2, 2, 3, 3]):
        """Build a deep denoising autoencoder w/ tied weights.

        Parameters
        ----------
        input_shape : list, optional
            Description
        n_filters : list, optional
            Description
        filter_sizes : list, optional
            Description

        Raises
        ------
        ValueError
            Description
        """
        # %%
        # input to the network

        self.filters = n_filters

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')

        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        current_input = self.inputs

        # Optionally apply denoising autoencoder
        net = tf.cond(self.is_training, lambda: corrupt(current_input), lambda: current_input)

        # ENCODER
        net = self.conv2d_relu(net, layer_index=0, filter_size=2, output_channels=64)
        net = self.conv2d_relu(net, layer_index=1, filter_size=3, output_channels=128)
        net = self.max_pool(net, layer_index=1)

        net = self.conv2d_relu(net, layer_index=2, filter_size=2, output_channels=128)
        net = self.conv2d_relu(net, layer_index=3, filter_size=3, output_channels=256)
        net = self.max_pool(net, layer_index=3)

        print("current input shape", net.get_shape())

        self.encoder.reverse()
        self.shapes.reverse()
        self.pooling_indices.reverse()

        # DECODER
        net = unpool_layer2x2_batch(net, self.pooling_indices[0])
        net = self.conv2d_transposed_relu(net, self.shapes[0], layer_index=0)
        net = self.conv2d_transposed_relu(net, self.shapes[1], layer_index=1)

        net = unpool_layer2x2_batch(net, self.pooling_indices[1])
        net = self.conv2d_transposed_relu(net, self.shapes[2], layer_index=2)
        net = self.conv2d_transposed_relu(net, self.shapes[3], layer_index=3)

        net = tf.sigmoid(net)

        self.segmentation_result = net  # [batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS]

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))

        # MSE loss
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)

        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

    def conv2d_relu(self, input, layer_index, filter_size, output_channels):
        number_of_channels = input.get_shape().as_list()[3]
        self.shapes.append(input.get_shape().as_list())
        print(input.get_shape().as_list())
        W = tf.get_variable('W' + str(layer_index),
                            shape=(filter_size, filter_size, number_of_channels, output_channels))
        b = tf.Variable(tf.zeros([output_channels]))
        self.encoder.append(W)
        output = lrelu(tf.add(tf.nn.conv2d(input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        return output

    def conv2d_transposed_relu(self, input, shape, layer_index):
        W = self.encoder[layer_index]
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                input, W,
                tf.pack([tf.shape(self.inputs)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        return output

    def max_pool(self, input, layer_index):
        current_input, argmax = tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1],
                                                           strides=[1, 2, 2, 1], padding='SAME',
                                                           name='pool{}'.format(layer_index + 1))
        self.pooling_indices.append(argmax)
        return current_input
