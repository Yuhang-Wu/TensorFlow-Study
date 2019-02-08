import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import GenerateDS_4_5
import Forward_4_5

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    data_x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    true_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

    X, Y, Y_color = GenerateDS_4_5.generateds()

    y_hat = Forward_4_5.forward(data_x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, 300 / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

    # loss function
    loss_mse = tf.reduce_mean(tf.square(y_hat - true_y))
    loss_total = tf.add(loss_mse, tf.add_n(tf.get_collection('losses')))

    # backpropagation
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(STEPS):
            start = (i * BATCH_SIZE) % 300
            end = start + BATCH_SIZE

            sess.run(train_step, feed_dict={
                data_x: X[start:end], true_y: Y[start:end]})

        # xx 与 yy都在 -3 到 3 之间以步长为 0.01 生成二维网格坐标点
        xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
        # 将xx，yy拉直，并合并成一个2列的矩阵，得到一个网格坐标点的集合
        grid = np.c_[xx.ravel(), yy.ravel()]
        # 将网格坐标点feed into NN， probs为输出
        probs = sess.run(y_hat, feed_dict={data_x: grid})
        print(probs.shape)
        # probs 的shape调整成 xx 的样子
        probs = probs.reshape(xx.shape)
        print(probs.shape)

    plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__ == '__main__':
    backward()
