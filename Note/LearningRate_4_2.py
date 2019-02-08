import tensorflow as tf


def SimpleExample():
    """
     define loss function: loss = (w+1)^2;
     initialize w = 5;
     backpropagation find optimal w;
    """

    w = tf.Variable(tf.constant(5, dtype=tf.float32))

    # define loss function
    loss_degree = tf.square(w + 1)

    # backpropagation
    # train_setp = tf.train.GradientDescentOptimizer(0.2).minimize(loss_degree)
    # train_setp = tf.train.GradientDescentOptimizer(1).minimize(loss_degree)
    train_setp = tf.train.GradientDescentOptimizer(
        0.0001).minimize(loss_degree)

    # training model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(40):
            sess.run(train_setp)
            ws = sess.run(w)
            loss = sess.run(loss_degree)

            print(ws, loss)


def ExponentialDecay_LearningRate():
    """ Dynamic learning rate"""

    LEARNING_BASE = 0.1  # initialize learning rate
    LEARNING_RATE_DECAY = 0.99
    # feed多少轮 BATCH_SIZE之后，update learning rate, 一般设为：总样本数 / BATCH_SIZE
    LEARNING_RATE_STEP = 1

    # Counter: 计算一共运行多少轮 BATCH_SIZE, init = 0, not trainable
    global_step = tf.Variable(0, trainable=False)

    # defien ExponentialDecay_LearningRate
    learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_BASE, global_step=global_step,
                                               decay_steps=LEARNING_RATE_STEP, decay_rate=LEARNING_RATE_DECAY, staircase=True)
    # trainable parameters
    w = tf.Variable(tf.constant(5, dtype=tf.float32))

    # loss function
    loss_func = tf.square(w + 1)

    # backpropagation: OPTIMIZE
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss_func)

    # training model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(40):
            sess.run(train_step)
            learning_rate_val = sess.run(learning_rate)
            global_step_val = sess.run(global_step)
            w_val = sess.run(w)
            loss_val = sess.run(loss_func)

            print(learning_rate_val, global_step_val, w_val, loss_val)


ExponentialDecay_LearningRate()
