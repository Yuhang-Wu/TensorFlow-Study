import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BATCH_SIZE = 30
SEED = 2

rdm = np.random.RandomState(SEED)
X = rdm.randn(300, 2)
X.shape
Y = [int(x1 * x1 + x2 * x2 < 2) for (x1, x2) in X]  # [1, 2, 3,, ...]
Y_color = [['red' if y else 'blue'] for y in Y]

# Explaination for vstack:
# https://blog.csdn.net/csdn15698845876/article/details/73380803
X = np.vstack(X).reshape(-1, 2)
X.shape
Y = np.vstack(Y).reshape(-1, 1)  # [[1], [2], [3], ...]

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_color))
plt.show()


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape=shape), dtype=tf.float32)
    tf.add_to_collection(
        'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


data_x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
true_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.add(tf.matmul(data_x, w1), b1))

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y_hat = tf.add(tf.matmul(y1, w2), b2)

# loss function
loss_mse = tf.reduce_mean(tf.square(y_hat - true_y))
loss_total = tf.add(loss_mse, tf.add_n(tf.get_collection('losses')))

# backpropagation without regularization
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)  # 正则化


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    STEPS = 40000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 300
        end = start + BATCH_SIZE

        sess.run(train_step, feed_dict={
            data_x: X[start:end], true_y: Y[start:end]})

        # if i % 2000 == 0:
        #     loss = sess.run(loss_total, feed_dict={true_y: Y, data_x: X})
        #     print("After %d training step, total loss is %f" % (i, loss))

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
