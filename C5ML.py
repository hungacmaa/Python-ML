# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 08:11:05 2022

@author: Admin
"""

# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# # represents the heights of a group of people in meters
# heights = [[1.6], [1.65], [1.7], [1.73], [1.8]]
# # represents the weights of a group of people in kgs
# weights = [[60], [65], [72.3], [75], [80]]
# plt.title('Weights plotted against heights')
# plt.xlabel('Heights in meters')
# plt.ylabel('Weights in kilograms')
# plt.plot(heights, weights, 'k.')
# # axis range for x and y
# plt.axis([1.5, 1.85, 50, 90])
# plt.grid(True)

# # Create and fit the model
# model = LinearRegression()
# model.fit(X=heights, y=weights)
# # make prediction
# weight = model.predict([[1.60]])[0][0]
# print(round(weight,2)) # 76.04

# import matplotlib.pyplot as plt
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# #---copy from dataset into a 2-d list---
# X = []
# for target in range(2):
#     X.append([[], []])
#     for i in range(len(cancer.data)): # target is 0 or 1
#         if cancer.target[i] == target:
#             X[target][0].append(cancer.data[i][0]) # first feature - mean radius
#             X[target][1].append(cancer.data[i][1]) # second feature — mean texture
# colours = ("r", "b") # r: malignant, b: benign
# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111)
# for target in range(2):
#     ax.scatter(X[target][0],
#                X[target][1],
#                c=colours[target])
# ax.set_xlabel("mean radius")
# ax.set_ylabel("mean texture")
# plt.show()

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# #---copy from dataset into a 2-d array---
# X = []
# for target in range(2):
#     X.append([[], [], []])
#     for i in range(len(cancer.data)): # target is 0,1
#         if cancer.target[i] == target:
#             X[target][0].append(cancer.data[i][0])
#             X[target][1].append(cancer.data[i][1])
#             X[target][2].append(cancer.data[i][2])
# colours = ("r", "b") # r: malignant, b: benign
# fig = plt.figure(figsize=(18,15))
# ax = fig.add_subplot(111, projection='3d')
# for target in range(2):
#     ax.scatter(X[target][0],
#                X[target][1],
#                X[target][2],
#                c=colours[target])
# ax.set_xlabel("mean radius")
# ax.set_ylabel("mean texture")
# ax.set_zlabel("mean perimeter")
# plt.show()

# import pandas as pd
# import numpy as np
# import seaborn as sns; sns.set(font_scale=1.2)
# import matplotlib.pyplot as plt
# data = pd.read_csv('svm.csv')
# print(data)
# sns.lmplot('x1', 'x2',
#            data=data,
#            hue='r',
#            palette='Set1',
#            fit_reg=False,
#            scatter_kws={"s": 50}); 

import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)
data = pd.read_csv('house_sizes_prices_svm.csv')
sns.lmplot('size', 'price',
            data=data,
            hue='sold',
            palette='Set2',
            fit_reg=False,
            scatter_kws={"s": 50});

X = data[['size','price']].values
y = np.where(data['sold']=='y', 1, 0) #--1 for Y and 0 for N---
model = svm.SVC(kernel='linear').fit(X, y)
#---min and max for the first feature---
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#---min and max for the second feature---
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#---step size in the mesh---
h = (x_max / x_min) / 20
#---make predictions for each of the points in xx,yy---
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#---draw the result using a color plot---
Z = Z.reshape(xx.shape)
# plt.contourf(xx, yy, Z, cmap=plt.cm.Blues, alpha=0.3)
# plt.xlabel('Size of house')
# plt.ylabel('Asking price (1000s)')
# plt.title("Size of Houses and Their Asking Prices")

def will_it_sell(size, price):
    if(model.predict([[size, price]]))==0:
        print('Will not sell!')
    else:
        print('Will sell!')
#---do some prediction---
will_it_sell(2500, 400) # Will not sell!
will_it_sell(2500, 200) # Will sell!