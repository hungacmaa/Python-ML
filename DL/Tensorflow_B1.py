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

# <<<<<<< HEAD
a = tf.constant([[1, 2, 3], 
                 [4, 5, 6]], 
                tf.float32)
b = tf.constant([[1, 2, 3, 4], 
                 [5, 6, 7, 8], 
                 [9, 10, 11, 12]], 
                tf.float32)
with tf.Session() as sess:
    print(sess.run(tf.matmul(a, b)))


# BAI 1
data = [10, 2, 1, 0.5, 0, -0.5, -1., -2., -10.]

# tanh and sigmoid
E = tf.nn.tanh(data)
print(sess.run(E))
print(sess.run(tf.nn.sigmoid(data)))

# ReLU
R = tf.nn.relu6(data)
print(sess.run(R))

R = tf.nn.relu(data)
print(sess.run(R))

# BAI 3
#Import required packages
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# Getting the data ready
# Generate train dummy data for 1000 Students and dummy test for 500
#Columns :Age, Hours of Study &Avg Previous test scores
np.random.seed(2018) #Setting seed for reproducibility
train_data, test_data = np.random.random((1000, 3)), np.random.random((500, 3))
#Generate dummy results for 1000 students : Whether Passed (1) or Failed (0)
labels = np.random.randint(2, size=(1000, 1))
#Defining the model structure with the required layers, # of neurons, activation function and optimizers
model = Sequential()
model.add(Dense(5, input_dim=3, activation='relu')) # layer 1 dung 5 neuron, activation func dung relu (0-10)
model.add(Dense(4, activation='relu')) # layer 2 dung 4 neuron, activation func dung relu (0-10)
model.add(Dense(1, activation='sigmoid')) # layer cuoi, dau ra tra ve 1 nhan (1 neuron), af dung sigmoid (0-1)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train the model and make predictions
model.fit(train_data, labels, epochs=10, batch_size=32)
#Make predictions from the trained model
predictions = model.predict(test_data)

# =======
E = tf.nn.tanh([10,1.])
sess.run(E)
sess.close()
# >>>>>>> 655ecc5f52ec1657702ee9a423abb3c189b27008
