# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 07:21:34 2022

@author: Admin
"""

import tensorflow.compat.v1 as tf
x = tf.constant(5, tf.float32)
y = tf.constant([5], tf.float32)
z = tf.constant([5, 3, 4], tf.float32)
t = tf.constant([[5, 3, 4, 6], [2, 3, 4, 7]], tf.float32)
u = tf.constant([[[5, 3, 4, 6], [2, 3, 4, 0]]], tf.float32)
v = tf.constant([[[5, 3, 4, 6], [2, 3, 4, 0]],
                 [[5, 3, 4, 6], [2, 3, 4, 0]],
                 [[5, 3, 4, 6], [2, 3, 4, 0]]],
                tf.float32)
print(x)
print(y)
print(z)
print(t)
print(u)
print(v)

# =================================
tf.compat.v1.disable_eager_execution()
x1 = tf.Variable(5.3, tf.float32)
x2 = tf.Variable(4.3, tf.float32)
x = tf.multiply(x1, x2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    t = sess.run(x)
    print(t)
print(x)
# ==================================
tf.compat.v1.disable_eager_execution()
x1 = tf.Variable([[5.3, 4.5, 6.0],
                  [4.3, 4.3, 7.0]],
                 tf.float32)
x2 = tf.Variable([[4.3, 4.3, 7.0],
                  [5.3, 4.5, 6.0], ],
                 tf.float32)
x = tf.multiply(x1, x2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    t = sess.run(x)
    print(t)


# =======
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, None)
y = tf.add(x, x)
with tf.Session() as sess:
    x_data = 5
    result = sess.run(y, feed_dict={x: x_data})
    print(result)
# =========
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, [None, 3])
y = tf.add(x, x)
with tf.Session() as sess:
    x_data = [[1.5, 2.0, 3.3]]
    result = sess.run(y, feed_dict={x: x_data})
    print(result)
# ========
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, [None, None, 3])
y = tf.add(x, x)
with tf.Session() as sess:
    x_data = [[[1, 2, 3]]]
    result = sess.run(y, feed_dict={x: x_data})
    print(result)
# ======
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, [None, 4, 3])
y = tf.add(x, x)
with tf.Session() as sess:
    x_data = [[[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]]]
    result = sess.run(y, feed_dict={x: x_data})
    print(result)
# ========
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, [2, 4, 3])
y = tf.add(x, x)
with tf.Session() as sess:
    x_data = [[[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]],
              [[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]]]

    result = sess.run(y, feed_dict={x: x_data})
    print(result)
# ========
tf.compat.v1.disable_eager_execution()
x = tf.placeholder(tf.float32, [2, 4, 3])
y = tf.placeholder(tf.float32, [2, 4, 3])
z = tf.add(x, y)
u = tf.multiply(x, y)
with tf.Session() as sess:
    x_data = [[[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]],
              [[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]]]
    y_data = [[[1, 2, 3],
              [2, 3, 4],
              [2, 3, 5],
              [0, 1, 2]],
              [[1, 2, 3],
               [2, 3, 4],
               [2, 3, 5],
               [0, 1, 2]
               ]]
result1 = sess.run(z, feed_dict={x: x_data, y: y_data})
result2 = sess.run(u, feed_dict={x: x_data, y: y_data})
print("result1 =", result1)
print("result2 =", result2)
# ========