"""
 Tensor: Multidimensional array

 Scalar: s = 1, 2, 3, ...
 Vector: v = [1, 2, 3, ...]
 Matrix: m = [[1, 2, 3], [4, 5, 6], [7, 8, 9] , ...]
 Tensor: t = [[[[[...]]]]]
"""

import numpy as np
import tensorflow as tf

a = tf.constant([1.0, 2.0])
print(a)
b = tf.constant([3.0, 4.0])
result = a + b

"""
 "add:0":       node name
 shape=(2,0):   1-dimensional vector, length 2
 dtype=float32: data type
"""
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)


x1 = tf.constant([[1.0, 2.0]])
print(x1)
x2 = tf.constant([1.0, 2.0])
print(x2)
w = tf.constant([[3.0], [4.0]])
print(w)
y = tf.matmul(x, w)
print(y)
with tf.Session() as sess:
    print(sess.run(y))


def simpleNN():
    """ Simple two fully connect NN, no improvement"""
    # define inputs and parameter
    x = tf.constant([[0.7, 0.5]])  # x1=0.7, x2=0.5
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # feedforward graph
    h = tf.matmul(x, w1)
    y = tf.matmul(h, w2)

    # implement
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))


simpleNN()


def FeedOneDataPointNN():
    # define inputs and parameter
    x = tf.placeholder(tf.float32, shape=(1, 2))  # feed one data point
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # feedforward graph
    h = tf.matmul(x, w1)
    y = tf.matmul(h, w2)

    # implement
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y, feed_dict={x: [[0.7, 0.5]]}))


FeedOneDataPointNN()


def FeedMultipleDataNN():
    # define inputs and parameter
    # feed aultiple data points
    x = tf.placeholder(tf.float32, shape=(None, 2))
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # feedforward graph
    h = tf.matmul(x, w1)
    y = tf.matmul(h, w2)

    # implement
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4]]}))


FeedMultipleDataNN()
