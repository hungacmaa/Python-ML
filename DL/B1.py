# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:24:05 2022

@author: Admin
"""

import tensorflow.compat.v1 as tf
x = tf.constant(5.5)

y = tf.constant([5,4,3,2,1])

z = tf.constant([[5,4,3,2,1],
                 [5,4,3,2,1],
                 [5,4,3,2,1]])

t = tf.constant([[[5,4,3,2,1],
                  [5,4,3,2,1],
                  [5,4,3,2,1]],
                 [[5,4,3,2,1],
                  [5,4,3,2,1],
                  [5,4,3,2,1]],
                 [[5,4,3,2,1],
                  [5,4,3,2,1],
                  [5,4,3,2,1]]])
print(x)
print(y)
print(z)
print(t)

a = tf.constant([1,2,3,5,5])
b = tf.constant(2)
c = tf.add(a, z)
print(c)

x1 = tf.Variable(1)
x2 = tf.Variable(2)
z = tf.add(x1,x2)
print(z)

import tensorflow as tf
sess = tf.Session()
print(sess.run(z))