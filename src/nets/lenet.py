import tensorflow as tf
import nets.base_layers as layers

FILTER_SIZE = 5


def stack(network_input, network_input_depth, output_size):
    first_conv_stack = layers.conv2d(network_input, filters_size=FILTER_SIZE, input_depth=network_input_depth,
                                                  num_filters=6)
    first_conv_stack = layers.max_pool(first_conv_stack)
    first_conv_stack = layers.relu(first_conv_stack)
    
    second_conv_stack = layers.conv2d(first_conv_stack, filters_size=FILTER_SIZE,
                                                   input_depth=6, num_filters=16)
    second_conv_stack = layers.max_pool(second_conv_stack)
    second_conv_stack = layers.relu(second_conv_stack)
    
    flatten_layer = tf.reshape(second_conv_stack, [-1, 400])

    first_fully_connected = layers.fully_connected(flatten_layer, 400, 120)
    first_fully_connected = layers.relu(first_fully_connected)

    second_fully_connected = layers.fully_connected(first_fully_connected, 120, 84)
    second_fully_connected = layers.relu(second_fully_connected)

    return layers.fully_connected(second_fully_connected, 84, output_size)
