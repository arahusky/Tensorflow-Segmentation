import utils
from layer import Layer


class MaxPool2d(Layer):
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def create_layer(self, input):
        return utils.max_pool_2d(input, self.kernel_size)

    def create_layer_reversed(self, input):
        return utils.upsample_2d(input, self.kernel_size)

    def get_description(self):
        return "M{};".format(self.kernel_size)
