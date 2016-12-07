"""Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import io
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from libs.activations import lrelu
from libs.utils import corrupt

np.set_printoptions(threshold=np.nan)


@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad(op.inputs[0],
                                     op.outputs[0],
                                     grad,
                                     op.get_attr("ksize"),
                                     op.get_attr("strides"),
                                     padding=op.get_attr("padding"),
                                     data_format='NHWC')


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


class Network:
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1

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
        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS], name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')

        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        current_input = self.inputs

        # Optionally apply denoising autoencoder
        current_input = tf.cond(self.is_training, lambda: corrupt(current_input), lambda: current_input)

        # Build the encoder
        encoder = []
        shapes = []
        for layer_index, output_channels in enumerate(n_filters[1:]):
            number_of_channels = current_input.get_shape().as_list()[3]
            shapes.append(current_input.get_shape().as_list())

            W = tf.get_variable('W' + str(layer_index), shape=(filter_sizes[layer_index], filter_sizes[layer_index], number_of_channels, output_channels))
            b = tf.Variable(tf.zeros([output_channels]))
            encoder.append(W)
            output = lrelu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        # current_input, argmax_1 = tf.nn.max_pool_with_argmax(current_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # store the latent representation
        # z = current_input

        print("current input shape", current_input.get_shape())

        # current_input = unpool_layer2x2_batch(current_input, argmax_1)

        encoder.reverse()
        shapes.reverse()

        # Build the decoder using the same weights
        for layer_index, shape in enumerate(shapes):
            W = encoder[layer_index]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    current_input, W,
                    tf.pack([tf.shape(self.inputs)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        current_input = tf.sigmoid(current_input)

        self.segmentation_result = current_input  # [batch_size, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS]

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))

        # MSE loss
        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)


class Dataset:
    def __init__(self, folder='data28_28', batch_size=50):
        self.batch_size = batch_size

        self.train_inputs = []
        self.train_targets = []
        self.test_inputs = []
        self.test_targets = []

        train_files = []
        test_files = []
        with io.open(os.path.join(folder, 'train.txt'), 'r') as reader:
            train_files += reader.readlines()

        with io.open(os.path.join(folder, 'validation.txt'), 'r') as reader:
            train_files += reader.readlines()

        with io.open(os.path.join(folder, 'test.txt'), 'r') as reader:
            test_files += reader.readlines()

        for train_file in train_files:
            input_image, target_image = train_file.strip().split(' ')
            train_image = np.array(Image.open(os.path.join(folder, input_image)).convert('L'))  # .convert('L')) -> grayscale (1-channel)
            train_image = np.multiply(train_image, 1.0 / 255)
            self.train_inputs.append(train_image)
            self.train_targets.append(np.array(Image.open(os.path.join(folder, target_image)).convert('1')).astype(np.float32))  # .convert('1')) -> binary

        for test_file in test_files:
            input_image, target_image = test_file.strip().split(' ')
            test_image = np.array(Image.open(os.path.join(folder, input_image)).convert('L'))
            test_image = np.multiply(test_image, 1.0 / 255)
            self.test_inputs.append(test_image)
            self.test_targets.append(np.array(Image.open(os.path.join(folder, target_image)).convert('1')).astype(np.float32))

        self.pointer = 0

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size

        return np.array(inputs), np.array(targets)


def test_mnist():
    """Test the convolutional autoencder using MNIST."""
    # %%

    dataset = Dataset()

    inputs, targets = dataset.next_batch()
    print(inputs.shape, targets.shape)
    # print(targets[0].astype(np.float32))

    # Image.fromarray(targets[0] * 255).show()

    # Image.fromarray(targets[0], 'RGB').show()

    # load MNIST as before
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mean_img = np.mean(dataset.train_inputs)

    network = Network()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Fit all training data
        batch_size = 100
        n_epochs = 20
        for epoch_i in range(n_epochs):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                start = time.time()
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs, (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets, (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                # train = np.array([img - mean_img for img in batch_inputs]).reshape((dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, network.IMAGE_CHANNELS))
                cost, _ = sess.run([network.cost, network.train_op], feed_dict={network.inputs: batch_inputs, network.targets: batch_targets, network.is_training: True})
                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(epoch_i * dataset.num_batches_in_epoch() + batch_i,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost, end - start))

        # Plot example reconstructions
        n_examples = 20
        test_inputs, test_targets = dataset.test_inputs[:n_examples], dataset.test_targets[:n_examples]

        test_segmentation = sess.run(network.segmentation_result, feed_dict={network.inputs: np.reshape(test_inputs, [n_examples, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})
        fig, axs = plt.subplots(4, n_examples, figsize=(n_examples, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
            axs[1][example_i].imshow(test_targets[example_i], cmap='gray')
            axs[2][example_i].imshow(np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]), cmap='gray')

            test_image_thresholded = np.array([0 if x < 0.5 else 1 for x in test_segmentation[example_i].flatten()])
            axs[3][example_i].imshow(np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]), cmap='gray')
        # fig.show()
        # plt.draw()

        plt.savefig('figure{}.jpg'.format(batch_i + epoch_i * dataset.num_batches_in_epoch()))


if __name__ == '__main__':
    test_mnist()
