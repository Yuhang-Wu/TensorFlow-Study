import tensorflow as tf


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


def forward(data_x, regularizer):

    w1 = get_weight([2, 11], 0.01)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.add(tf.matmul(data_x, w1), b1))

    w2 = get_weight([11, 1], 0.01)
    b2 = get_bias([1])
    y_hat = tf.add(tf.matmul(y1, w2), b2)

    return y_hat
