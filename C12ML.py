# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 20:51:30 2022

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
# df.info()

#--- clean data ----
#---check for null values---
# print("Nulls")
# print("=====")
# print(df.isnull().sum())


df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)

df.fillna(df.mean(numeric_only=True), inplace = True) # replace NaN with the mean
# print(df.mean(numeric_only=True))
#---check for 0s---
# print("0s")
# print("==")
# print(df.eq(0).sum())

corr = df.corr()
# print(corr)

# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(df.columns),1)
# ax.set_xticks(ticks)
# ax.set_xticklabels(df.columns)
# plt.xticks(rotation = 90)
# ax.set_yticks(ticks)
# ax.set_yticklabels(df.columns)
# #---print the correlation factor---
# for i in range(df.shape[1]):
#     for j in range(9):
#         text = ax.text(j, i, round(corr.iloc[i][j],2), ha="center", va="center", color="w")
# plt.show()

#---get the top four features that has the highest correlation---
# print(df.corr().nlargest(4, 'Outcome').index)
#---print the top 4 correlation values---
# print(df.corr().nlargest(4, 'Outcome').values[:,8])

# ---- before compare the algorithms, we create a result array --------
result = []

# ---- logistic regression ----
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#---features---
X = df[['Glucose','BMI','Age']]
# print(X)
#---label---
y = df.iloc[:,8]
# print(y)
log_regress = linear_model.LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, y, cv=10, scoring='accuracy').mean()
# print(log_regress_score)
result.append(log_regress_score)

# ---- KNN ----
from sklearn.neighbors import KNeighborsClassifier
#---empty list that will hold cv (cross-validates) scores---
cv_scores = []
#---number of folds---
folds = 10
#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
# print(len(ks))
#---perform k-fold cross validation---
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
    cv_scores.append(score)
#---get the maximum score---
knn_score = max(cv_scores)
#---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
# print(f'The optimal number of neighbors is {optimal_k}')
# print(knn_score)
result.append(knn_score)

# ---- Support vector machines ----

# ---- used linear kernel ----
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y, cv=10, scoring='accuracy').mean()
# print(linear_svm_score)
result.append(linear_svm_score)

# ---- used RBF kernel ----
rbf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
# print(rbf_score)
result.append(rbf_score)

algorithms = ["Logistic Regression", "K Nearest Neighbors", "SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result,index = algorithms)
cv_mean.columns=["Accuracy"]

knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X.values, y)

import pickle
#---save the model to disk---
filename = 'diabetes.sav'
#---write to the file using write and binary mode---
pickle.dump(knn, open(filename, 'wb'))
#---load the model from disk---
loaded_model = pickle.load(open(filename, 'rb'))

Glucose = 65
BMI = 70
Age = 50
# prediction = loaded_model.predict([[Glucose, BMI, Age]])
# print(prediction)
# if (prediction[0]==0):
#     print("Non-diabetic")
# else:
#     print("Diabetic")

# proba = loaded_model.predict_proba([[Glucose, BMI, Age]])
# print(proba)
# print("Confidence: " + str(round(np.amax(proba[0]) * 100 ,2)) + "%")
