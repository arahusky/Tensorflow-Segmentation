import tensorflow as tf
import numpy as np

# Auto format padding
def autoformat_padding(padding):
    if padding in ['same', 'SAME', 'valid', 'VALID']:
        return str.upper(padding)
    else:
        raise Exception("Unknown padding! Accepted values: 'same', 'valid'.")


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")

# Auto format kernel
def autoformat_kernel_2d(strides):
    if isinstance(strides, int):
        return [1, strides, strides, 1]
    elif isinstance(strides, (tuple, list)):
        if len(strides) == 2:
            return [1, strides[0], strides[1], 1]
        elif len(strides) == 4:
            return [strides[0], strides[1], strides[2], strides[3]]
        else:
            raise Exception("strides length error: " + str(len(strides))
                            + ", only a length of 2 or 4 is supported.")
    else:
        raise Exception("strides format error: " + str(type(strides)))


def max_pool_2d(incoming, kernel_size, strides=None, padding='same',
                name="MaxPool2D"):
    """ Max Pooling 2D.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer.
        kernel_size: 'int` or `list of int`. Pooling kernel size.
        strides: 'int` or `list of int`. Strides of conv operation.
            Default: same as kernel_size.
        padding: `str` from `"same", "valid"`. Padding algo to use.
            Default: 'same'.
        name: A name for this layer (optional). Default: 'MaxPool2D'.
    Attributes:
        scope: `Scope`. This layer scope.
    """
    input_shape = get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"

    kernel = autoformat_kernel_2d(kernel_size)
    strides = autoformat_kernel_2d(strides) if strides else kernel
    padding = autoformat_padding(padding)

    with tf.name_scope(name) as scope:
        inference = tf.nn.max_pool(incoming, kernel, strides, padding)

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    # tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def upsample_2d(incoming, kernel_size, name="UpSample2D"):
    """ UpSample 2D.
    Input:
        4-D Tensor [batch, height, width, in_channels].
    Output:
        4-D Tensor [batch, pooled height, pooled width, in_channels].
    Arguments:
        incoming: `Tensor`. Incoming 4-D Layer to upsample.
        kernel_size: 'int` or `list of int`. Upsampling kernel size.
        name: A name for this layer (optional). Default: 'UpSample2D'.
    Attributes:
        scope: `Scope`. This layer scope.
    """
    input_shape = get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    kernel = autoformat_kernel_2d(kernel_size)

    with tf.name_scope(name) as scope:
        inference = tf.image.resize_nearest_neighbor(
            incoming, size=input_shape[1:3] * tf.constant(kernel[1:3]))
        inference.set_shape((None, input_shape[1] * kernel[1],
                             input_shape[2] * kernel[2], None))

    # Add attributes to Tensor to easy access weights
    inference.scope = scope

    # Track output tensor.
    # tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference
