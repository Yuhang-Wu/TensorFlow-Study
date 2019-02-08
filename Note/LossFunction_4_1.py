# 预测伤脾日销量 y，x1 和 x2是影响销量的两个因素

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 8
SEED = 23455
learning_rate = 0.001

COST = 1
PROFIT = 9

rdm = np.random.RandomState(SEED)
X = rdm.rand(32, 2)  # 32 * 2 matrix
Y = [[x1 + x2 + (rdm.rand() / 10 - 0.05)] for (x1, x2) in X]


def plot_loss(loss):
    p = np.arange(len(loss))
    plt.plot(p, loss)
    plt.title('Loss')
    plt.show()


# NN inputs and parameters
data_x = tf.placeholder(shape=(None, 2), dtype=tf.float32)
true_y = tf.placeholder(shape=(None, 1), dtype=tf.float32)
weight = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y_hat = tf.matmul(data_x, weight)

# loss function and optimization
# loss_mse = tf.reduce_mean(tf.square(y_hat - true_y))
loss_diy = tf.reduce_sum(tf.where(tf.greater(
    y_hat, true_y), COST * (y_hat - true_y), PROFIT * (true_y - y_hat)))
train_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss_diy)

# training model
losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS = 20000

    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE

        _, loss = sess.run([train_step, loss_diy], feed_dict={
                           data_x: X[start:end], true_y: Y[start:end]})
        losses.append(loss)

    print(sess.run(weight, feed_dict={data_x: X}))
plot_loss(losses)
