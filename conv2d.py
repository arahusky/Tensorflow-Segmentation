import tensorflow as tf
from layer import Layer
from libs.activations import lrelu


class Conv2d(Layer):
    # global things...
    layer_index = 0

    def __init__(self, kernel_size, strides, output_channels):
        self.kernel_size = kernel_size
        self.strides = strides
        self.output_channels = output_channels

    @staticmethod
    def reverse_global_variables():
        Conv2d.layer_index = 0

    def create_layer(self, input):
        number_of_channels = input.get_shape().as_list()[3]
        self.shape = input.get_shape().as_list()
        print(input.get_shape().as_list())
        W = tf.get_variable('W' + str(Conv2d.layer_index),
                            shape=(self.kernel_size, self.kernel_size, number_of_channels, self.output_channels))
        b = tf.Variable(tf.zeros([self.output_channels]))
        self.encoder = W
        Conv2d.layer_index += 1

        output = lrelu(tf.add(tf.nn.conv2d(input, W, strides=self.strides, padding='SAME'), b))
        return output

    def create_layer_reversed(self, input):
        print('convd2_transposed: {}, {}'.format(self.shape, tf.shape(input)[0]))
        # W = self.encoder[layer_index]
        W = tf.get_variable("W_deconv" + str(Conv2d.layer_index), shape=self.encoder.get_shape())
        b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                input, W,
                tf.pack([100, self.shape[1], self.shape[2], self.shape[3]]),
                strides=self.strides, padding='SAME'), b))

        Conv2d.layer_index += 1
        return output

    def get_description(self):
        return "C:{}K:{}".format(self.output_channels, self.kernel_size)
