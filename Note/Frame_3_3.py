""" Simple NN with Optimizer """
import tensorflow as tf
import numpy as np

# how many training data feeded in to NN at once
BATCH_SIZE = 8
seed = 23455

# generate random number
rng = np.random.RandomState(seed)

# training dataset with 32 * 2 matrix: [x0, x1]
X = rng.rand(32, 2)
X.shape

# set true labels with condition
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]  # list of lists 32 * 1 matrix

# 1 define input, parameter, outputs and graph
data_x = tf.placeholder(tf.float32, shape=(None, 2))
true_y = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal(shape=[2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal(shape=[3, 1], stddev=1, seed=1))

# hidden layer and output layer
h = tf.matmul(data_x, w1)
y_hat = tf.matmul(h, w2)


# 2 define loss function
loss = tf.reduce_mean(tf.square(true_y - y_hat))
# Optimizer
train_step = tf.train.GradientDescentOptimizer(
    learning_rate=0.01).minimize(loss)
# train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 3 iterate train model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # print the current parameter value before optimize
    print("w1: ", sess.run(w1, feed_dict={data_x: X}))
    print("w2: ", sess.run(w2, feed_dict={data_x: X}), '\n')

    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE

        sess.run(train_step, feed_dict={
                 data_x: X[start:end], true_y: Y[start:end]})

        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={data_x: X, true_y: Y})
            print("After %d training steps, loss on all data is %g" %
                  (i, total_loss))

    print("w1: ", sess.run(w1, feed_dict={data_x: X}))
    print("w2: ", sess.run(w2, feed_dict={data_x: X}), '\n')
