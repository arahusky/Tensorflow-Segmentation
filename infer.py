import io
import os

import tensorflow as tf
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2

import matplotlib.pyplot as plt

if __name__ == '__main__':

    layers = []
    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64))
    layers.append(MaxPool2d(kernel_size=2))

    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64))
    layers.append(MaxPool2d(kernel_size=2))

    layers.append(Conv2d(kernel_size=7, strides=[1, 2, 2, 1], output_channels=64))
    layers.append(Conv2d(kernel_size=7, strides=[1, 1, 1, 1], output_channels=64))
    layers.append(MaxPool2d(kernel_size=2))


    network = convolutional_autoencoder.Network(layers)


    input_image = '/home/arahusky/Desktop/aja.jpg'
    checkpoint = 'save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2016-12-14_162230/'

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))


        image = np.array(cv2.imread(input_image, 0))  # load grayscale
        image = cv2.resize(image, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
        image = np.multiply(image, 1.0/255)
        cv2.imshow('image', image)
        cv2.waitKey(0)

        print(image.shape)

        segmentation = sess.run(network.segmentation_result, feed_dict={
            network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

        print(segmentation[0].shape)
        fig, axs = plt.subplots(2, 1, figsize=(1 * 3, 10))
        axs[0].imshow(image, cmap='gray')
        axs[1].imshow(np.reshape(segmentation[0],(128,128)), cmap='gray')

        plt.show()
        plt.savefig('result.jpg')
        plt.waitforbuttonpress()


