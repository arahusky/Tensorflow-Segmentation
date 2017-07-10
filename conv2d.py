import tensorflow as tf

import utils
from layer import Layer
from libs.activations import lrelu


class Conv2d(Layer):
    # global things...
    layer_index = 0

    def __init__(self, kernel_size, strides, output_channels, name):
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_channels = output_channels
        self.name = name

    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0

    def create_layer(self, input):
        # print('convd2: input_shape: {}'.format(utils.get_incoming_shape(input)))
        self.input_shape = utils.get_incoming_shape(input)
        number_of_input_channels = self.input_shape[3]

        with tf.variable_scope('conv', reuse=False):
            W = tf.get_variable('W{}'.format(self.name[-3:]),
                                shape=(self.kernel_size, self.kernel_size, number_of_input_channels, self.output_channels))
            b = tf.Variable(tf.zeros([self.output_channels]))
        self.encoder_matrix = W
        Conv2d.layer_index += 1

        output = tf.nn.conv2d(input, W, strides=self.strides, padding='SAME')

        # print('convd2: output_shape: {}'.format(utils.get_incoming_shape(output)))

        output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))

        return output

    def create_layer_reversed(self, input, prev_layer=None):
        # print('convd2_transposed: input_shape: {}'.format(utils.get_incoming_shape(input)))
        # W = self.encoder[layer_index]
        with tf.variable_scope('conv', reuse=True):
            W = tf.get_variable('W{}'.format(self.name[-3:]))
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

        # if self.strides==[1, 1, 1, 1]:
        #     print('Now')
        #     output = lrelu(tf.add(
        #         tf.nn.conv2d(input, W,strides=self.strides, padding='SAME'), b))
        # else:
        #     print('1Now1')
        output = tf.nn.conv2d_transpose(
            input, W,
            tf.stack([tf.shape(input)[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]]),
            strides=self.strides, padding='SAME')

        Conv2d.layer_index += 1
        output.set_shape([None, self.input_shape[1], self.input_shape[2], self.input_shape[3]])

        output = lrelu(tf.add(tf.contrib.layers.batch_norm(output), b))
        # print('convd2_transposed: output_shape: {}'.format(utils.get_incoming_shape(output)))

        return output

    def get_description(self):
        return "C{},{},{}".format(self.kernel_size, self.output_channels, self.strides[1])
