import tensorflow as tf
from nets import base_layers as layers

FIRST_STAGE = {
    'depth': 108,
    'filter_size': 5,
    'padding': 'SAME',
    'stride': (1, 1, 1, 1)
}
SECOND_STAGE = {
    'depth': 108,
    'filter_size': 5,
    'padding': 'SAME',
    'stride': (1, 1, 1, 1)
}
FULL_CONNECTION = {
    'hidden_units': 100,
}
FLATTEN_LAYER_SIZE = 108 * 8 * 8


def stack(network_input, network_input_depth, output_size):

    first_stage = layers.conv2d(network_input, FIRST_STAGE['filter_size'], FIRST_STAGE['depth'],
                                     network_input_depth, FIRST_STAGE['padding'], FIRST_STAGE['stride'])
    first_stage = layers.max_pool(first_stage)
    first_stage = layers.relu(first_stage)

    second_stage = layers.conv2d(first_stage, SECOND_STAGE['filter_size'], SECOND_STAGE['depth'],
                                      FIRST_STAGE['depth'], SECOND_STAGE['padding'], SECOND_STAGE['stride'])
    second_stage = layers.max_pool(second_stage)
    second_stage = layers.relu(second_stage)

    skip_layer = layers.max_pool(first_stage)
    skip_layer = tf.add(second_stage, skip_layer)
    skip_layer = tf.reshape(skip_layer, [-1, FLATTEN_LAYER_SIZE])

    fully_connected = layers.fully_connected(skip_layer, FLATTEN_LAYER_SIZE, FULL_CONNECTION['hidden_units'])
    classification = layers.fully_connected(fully_connected, FULL_CONNECTION['hidden_units'], output_size)

    return classification
