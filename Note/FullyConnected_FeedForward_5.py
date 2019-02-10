import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def Parameters(wShape, bShape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape=wShape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection(
            'losses', tf.cotrib.layers.l2_regularizer(regularizer)(w))

    b = tf.Variable(tf.zeros(bShape))

    return w, b


def Feedforward(x, regularizer):

    w1, b1 = Parameters([INPUT_NODE, LAYER1_NODE], [LAYER1_NODE], regularizer)
    layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2, b2 = Parameters([LAYER1_NODE, OUTPUT_NODE], [OUTPUT_NODE], regularizer)
    y_hat = tf.matmul(layer1, w2) + b2

    return y_hat
