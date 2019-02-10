import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import GenerateDS

GAMMA = 0.1
NUM_FEATURE = 2
NUM_ITER = 3000
learning_rate = 0.01


def ScatterPlot(data, labels, gamma):
    """ Plot the input data points with corresponding labels """

    fig, ax = plt.subplots(figsize=(16, 8))
    for i, label in enumerate(labels):
        plt.scatter(data[i, 0], data[i, 1], label=label,
                    color=['red' if label < 0 else 'green'])
    plt.axvline(gamma / 2, linestyle='--', color='indigo', alpha=0.3)
    plt.axvline(-gamma / 2, linestyle='--', color='indigo', alpha=0.3)
    plt.show()


def main():

    # generate data
    data, labels = GenerateDS.GenerateInputs(GAMMA)  # labels.reshape(-1, 1)
    labels_reshape = np.vstack(labels)
    ScatterPlot(data, labels, GAMMA)  # plot data with true labels

    # # Feedforward Graph
    # data_x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
    # true_y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    # weight = tf.Variable(tf.random_normal(shape=[2, 1]), dtype=tf.float32)
    #
    # y_hat = tf.reduce_sum(tf.multiply(data_x, weight))
    # y_predict = tf.sign(tf.sign(y_hat) + 0.00001)
    #
    # # backpropagation
    # check = tf.multiply(true_y, y_hat)
    # loss = tf.math.maximum(-true_y * y_hat, 0)
    #
    # train_step = None
    # update_weight = weight + learning_rate * true_y * tf.transpose(data_x)
    # if y_hat != true_y:
    #     train_step = tf.assign(weight, update_weight)
    #
    # # Execute model
    # losses = []
    # counter = 0
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     while True:
    #         print('Number of full pass:', counter + 1)
    #         counter += 1
    #         stopping = np.zeros(400)
    #
    #         for i in range(len(data)):
    #             _, l, cond = sess.run([train_step, loss, check],
    #                                   feed_dict={data_x: data[i].reshape(-1, 2),
    #                                              true_y: labels_reshape[i].reshape(-1, 1)})
    #             # print(stopping)
    #             losses.append(l)
    #             stopping[i] = cond
    #
    #         all_greater = False if False in [
    #             item > 0.0 for item in stopping] else True
    #
    #         if all_greater:
    #             break
    #
    #     w = weights.eval()
