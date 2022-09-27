# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 21:09:10 2022

@author: HungNguyen
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston

# load dữ liệu từ tập dữ liệu nhà ở ở boston
dataset = load_boston()
print(dataset.data)
print(dataset.feature_names) # xem tên các trường
print(dataset.DESCR) # xem mô tả tập dữ liệu
print(dataset.target) #giá của từng ngôi nhà

#chuyển tập dữ liệu về dạng dataframe
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head()

# thêm trường giá (target)
df['MEDV'] =dataset.target
df.head()

#cleaning data
df.info()

print(df.isnull().sum())

#feature selection
corr = df.corr()
print(corr['MEDV'])

#---get the top 3 features that has the highest correlation---
print(df.corr().abs().nlargest(3, 'MEDV').index)
#---print the top 3 correlation values---
print(df.corr().abs().nlargest(3, 'MEDV').values[:,13])

# plot
plt.scatter(df['LSTAT'], df['MEDV'], marker='o')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')

plt.scatter(df['RM'], df['MEDV'], marker='o')
plt.xlabel('RM')
plt.ylabel('MEDV')

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['LSTAT'],
df['RM'],
df['MEDV'],
c='b')
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
plt.show()

#training the model

x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']

from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size = 0.3, random_state=5)

print(x_train.shape)
print(Y_train.shape)

print(x_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, Y_train)

price_pred = model.predict(x_test)

print('R-Squared: %.4f' % model.score(x_test, Y_test))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, price_pred)
print(mse)
plt.scatter(Y_test, price_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices")

#Getting the Intercept and Coefficients
print(model.intercept_)
print(model.coef_)

print(model.predict([[30,5]]))

#Plotting the 3D Hyperplane
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_boston
dataset = load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['MEDV'] = dataset.target
x = pd.DataFrame(np.c_[df['LSTAT'], df['RM']], columns = ['LSTAT','RM'])
Y = df['MEDV']
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x['LSTAT'],
x['RM'],
Y,
c='b')
ax.set_xlabel("LSTAT")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")
#---create a meshgrid of all the values for LSTAT and RM---
x_surf = np.arange(0, 40, 1) #---for LSTAT---
y_surf = np.arange(0, 10, 1) #---for RM---
x_surf, y_surf = np.meshgrid(x_surf, y_surf)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, Y)
#---calculate z(MEDC) based on the model---
z = lambda x,y: (model.intercept_ + model.coef_[0] * x + model.coef_[1] * y)
ax.plot_surface(x_surf, y_surf, z(x_surf,y_surf),
rstride=1,
cstride=1,
color='None',
alpha = 0.4)
plt.show()

#Polynomial Regression
df = pd.read_csv('abc.csv')
plt.scatter(df.x,df.y)