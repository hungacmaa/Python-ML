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

fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
ax.set_xticklabels(df.columns)
plt.xticks(rotation = 90)
ax.set_yticklabels(df.columns)
ax.set_yticks(ticks)
#---print the correlation factor---
for i in range(df.shape[1]):
    for j in range(9):
        text = ax.text(j, i, round(corr.iloc[i][j],2),
                       ha="center", va="center", color="w")
plt.show()

print(df.corr().nlargest(4, 'Outcome').values[:,8])