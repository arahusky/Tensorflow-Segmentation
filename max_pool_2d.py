import tensorflow as tf

import utils
from layer import Layer

class MaxPool2d(Layer):
    def __init__(self, kernel_size, name, skip_connection=False):
        self.kernel_size = kernel_size
        self.name = name
        self.skip_connection = skip_connection

    def create_layer(self, input):
        return utils.max_pool_2d(input, self.kernel_size)

    def create_layer_reversed(self, input, prev_layer=None):
        if self.skip_connection:
            input = tf.add(input, prev_layer)

        return utils.upsample_2d(input, self.kernel_size)

    def get_description(self):
        return "M{}".format(self.kernel_size)
