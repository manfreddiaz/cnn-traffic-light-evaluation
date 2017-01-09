import tensorflow as tf
import nets.base_layers as pr

FIRST_DEPTH = 16
FILTER_SIZE = 3


def stack(network_input, network_input_depth, output_size):
    conv3_64 = pr.conv2d(network_input, filters_size=FILTER_SIZE, num_filters=FIRST_DEPTH, input_depth=network_input_depth, strides=[1,1,1,1], padding='SAME')
    conv3_64 = pr.conv2d(conv3_64, filters_size=FILTER_SIZE, num_filters=FIRST_DEPTH, input_depth=FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_64 = pr.relu(conv3_64)
    max1 = pr.max_pool(conv3_64)

    conv3_128 = pr.conv2d(max1, filters_size=FILTER_SIZE, num_filters=2*FIRST_DEPTH, input_depth=FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_128 = pr.conv2d(conv3_128, filters_size=FILTER_SIZE, num_filters=2*FIRST_DEPTH, input_depth=2*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_128 = pr.relu(conv3_128)
    max2 = pr.max_pool(conv3_128)

    conv3_256 = pr.conv2d(max2, filters_size=FILTER_SIZE, num_filters=4*FIRST_DEPTH, input_depth=2*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_256 = pr.conv2d(conv3_256, filters_size=FILTER_SIZE, num_filters=4*FIRST_DEPTH, input_depth=4*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_256 = pr.relu(conv3_256)
    max3 = pr.max_pool(conv3_256)

    conv3_512 = pr.conv2d(max3, filters_size=FILTER_SIZE, num_filters=8*FIRST_DEPTH, input_depth=4*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_512 = pr.conv2d(conv3_512, filters_size=FILTER_SIZE, num_filters=8*FIRST_DEPTH, input_depth=8*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_512 = pr.relu(conv3_512)
    max4 = pr.max_pool(conv3_512)

    conv3_512 = pr.conv2d(max4, filters_size=FILTER_SIZE, num_filters=8*FIRST_DEPTH, input_depth=8*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_512 = pr.conv2d(conv3_512, filters_size=FILTER_SIZE, num_filters=8*FIRST_DEPTH, input_depth=8*FIRST_DEPTH, strides=[1,1,1,1], padding='SAME')
    conv3_512 = pr.relu(conv3_512)
    max5 = pr.max_pool(conv3_512)

    print(max5.get_shape())
    flatten_layer = tf.reshape(max5, [-1, 8*FIRST_DEPTH])

    fully_connected = pr.fully_connected(flatten_layer, 8*FIRST_DEPTH, 16*FIRST_DEPTH)
    fully_connected = pr.fully_connected(fully_connected, 16*FIRST_DEPTH, 16*FIRST_DEPTH)
    fully_connected = pr.fully_connected(fully_connected, 16*FIRST_DEPTH, output_size)

    return fully_connected

def vgg_stack(input, input_depth, volume_depth):
    vgg_convolutional = pr.conv2d(input, filters_size=3, num_filters=volume_depth, input_depth=input_depth,
                         strides=[1, 1, 1, 1], padding='SAME')
    vgg_convolutional = pr.conv2d(vgg_convolutional, filters_size=3, num_filters=volume_depth, input_depth=volume_depth, strides=[1, 1, 1, 1], padding='SAME')
    vgg_convolutional = pr.relu(vgg_convolutional)
    return pr.max_pool(vgg_convolutional)
