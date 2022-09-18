# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:58:33 2022

@author: HungNguyen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../diabetes.csv")
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# cf = SelectKBest(score_func=chi2, k=4)
# fit = cf.fit(X, y)
# dfcolumns = pd.DataFrame(X.columns)
# dfscores = pd.DataFrame(fit.scores_)
# #concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)
# featureScores.columns = ['Specs','Score']  #naming the dataframe columns
# print(featureScores.nlargest(4,'Score'))  #print 10 best features

# from sklearn.ensemble import ExtraTreesClassifier
# import matplotlib.pyplot as plt
# model = ExtraTreesClassifier()
# model.fit(X,y)
# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
# #plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(5).plot(kind='barh')
# plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")