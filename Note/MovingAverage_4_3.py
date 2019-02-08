import tensorflow as tf

# 1. 定义变量及滑动平均类
# 定义一个32位浮点变狼，初始值为0。0 这个代码就是不断更新及优化w参数，滑动平均做了w的影子
w = tf.Variable(0, dtype=tf.float32)

# 定义 num_updates (NN迭代轮输)，初始值 0.0 不可被优化
global_step = tf.Variable(0, trainable=False)

# 实例化滑动平均类，删减率 0.99， 当前轮数 global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

# ema.apply 后的括号里是更新列表，每次运行sess.run(ema_op)时， 对更新列表中的元素求滑动平均均值
# 实际应用中使用 tf.trainable_variables()自动将所有待训练的参数汇总为列表
# ema_op = ema.apply([w])
ema_op = ema.apply(tf.trainable_variables())

# 查看不同迭代中变量取值的变化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 用ema.average(w) 取滑动个平均值
    print(sess.run([w, ema.average(w)]))

    # 参数w的值赋为1
    sess.run(tf.assign(w, 1))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    # 更新step和w的值，模拟出100轮迭代之后，参数w变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w, 10))
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    # 每次sess.run() 会更新一次w的滑动平均值
    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))

    sess.run(ema_op)
    print(sess.run([w, ema.average(w)]))
