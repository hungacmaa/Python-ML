# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 07:24:05 2022

@author: Admin
"""

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
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

sess = tf.Session()
print(sess.run(z))


#import thư viện
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# Khởi tạo biến x, y
x = tf.Variable(2, name = "x")
y = tf.Variable(3, name="y")

# Khởi tạo hằng số
z = tf.constant(2, name="z")

# Tạo hàm f
f = x*x*y + y + z 
f = tf.add(tf.multiply(y, tf.multiply(x, x)), tf.add(y, z)) # cũng có thể tạo hàm bằng những hàm có sẵn của tensorflow

# Tạo session tính toán
sess = tf.Session()
sess.run(x.initializer) # khởi tạo giá trị cho biến x
sess.run(y.initializer) # khởi tạo giá trị cho biến y
result = sess.run(f) # Chạy và lấy kết quả sau khi tính hàm f
print("Giá trị hàm f tính được với x=2 và y=3 là: "+str(result))
sess.close() # đóng phiên sau khi đã hoàn tất tính toán

# Cũng có thể thực hiện session bằng cách sau
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # result = f.eval()
    sess.run(f)
    print(result)
    
x1 = tf.placeholder(tf.float32, shape=None)
print(x1.shape)
f1 = tf.add(x1,x1)
with tf.Session() as sess:
    x_data = 5
    result = sess.run(f1,feed_dict={x1:x_data}) # cách truyền giá trị cho placeholder
print(result)

E = tf.nn.tanh([10,1.])
sess.run(E)
sess.close()