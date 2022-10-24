# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 08:21:00 2022

@author: Admin
"""

import pandas as pd
df = pd.read_csv("train.csv")
print("Shape of the Dataset:",df.shape)
#the head method displays the first 5 rows of the data
df.head(5)
store = pd.read_csv("store.csv")
print("Shape of the Dataset:",store.shape)
#Display the first 5 rows of data using the head method of pandas dataframe
store.head(5)
df_new = df.merge(store,on=["Store"], how="inner")
print(df_new.shape)
print("Distinct number of Stores :", len(df_new["Store"]. unique()))
print("Distinct number of Days :", len(df_new["Date"]. unique()))
print("Average daily sales of all stores : ",round(df_new["Sales"].mean(),2))
print(df_new.dtypes)
df_new["DayOfWeek"].value_counts()

import numpy as np
df_new['Date'] = pd.to_datetime(df_new['Date'], infer_datetime_format=True)
df_new["Month"] = df_new["Date"].dt.month
df_new["Quarter"] = df_new["Date"].dt.quarter
df_new["Year"] = df_new["Date"].dt.year
df_new["Day"] = df_new["Date"].dt.day
df_new["Week"] = df_new["Date"].dt.isocalendar().week
df_new["Season"] = np.where(df_new["Month"].isin([3,4,5]),"Spring",
np.where(df_new["Month"].isin([6,7,8]),
"Summer",
np.where(df_new["Month"].isin
([9,10,11]),"Fall",
np.where(df_new["Month"].isin
([12,1,2]),"Winter","None"))))
#Using the head command to view (only) the data and the newly engineered features
print(df_new[["Date","Year","Month","Day","Week","Quarter","Season"]].head())

#Import matplotlib, python most popular data visualizing library
import matplotlib.pyplot as plt
# %matplotlib inline
#Create a histogram to study the Daily Sales for the stores
plt.figure(figsize=(15,8))
plt.hist(df_new["Sales"])
plt.title("Histogram for Store Sales")
plt.xlabel("bins")
plt.xlabel("Frequency")
plt.show()

#Use the histogram function provided by the Pandas object
#The function returns a cross-tab histogram plot for all numeric columns in the data
df_new.hist(figsize=(20,10))
df_new.isnull().sum()/df_new.shape[0] * 100

#Replace nulls with the mode
df_new["CompetitionDistance"]=df_new["CompetitionDistance"].fillna(df_new["CompetitionDistance"].mode()[0])
#Double check if we still see nulls for the column
df_new["CompetitionDistance"].isnull().sum()/df_new.shape[0] * 100

import seaborn as sns #Seaborn is another powerful visualization library for Python
sns.set(style="whitegrid")
#Create the bar plot for Average Sales across different Seasons
ax = sns.barplot(x="Season", y="Sales", data=df_new)
#Create the bar plot for Average Sales across different Assortments
ax = sns.barplot(x="Assortment", y="Sales", data=df_new)
#Create the bar plot for Average Sales across different Store Types
ax = sns.barplot(x="StoreType", y="Sales", data=df_new)
ax = sns.barplot(x="Season", y="Sales", data=df_new,
estimator=np.size)
ax = sns.barplot(x="Assortment", y="Sales", data=df_new,
estimator=np.size)
ax = sns.barplot(x="StoreType", y="Sales", data=df_new,
estimator=np.size)
#Define a variable for each type of feature
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
target = ["Sales"]
numeric_columns = ["Customers","Open","Promo","Promo2",
"StateHoliday","SchoolHoliday","CompetitionDistance"]
categorical_columns = ["DayOfWeek","Quarter","Month","Year",
"StoreType","Assortment","Season"]
#Define a function that will intake the raw dataframe and the column name and return a one hot encoded DF
def create_ohe(df, col):
    le = LabelEncoder()
    a=le.fit_transform(df_new[col]).reshape(-1,1)
    ohe = OneHotEncoder(sparse=False)
    column_names = [col+ "_"+ str(i) for i in le.classes_]
    return(pd.DataFrame(ohe.fit_transform(a),columns =column_names))
#Since the above function converts the column, one at a time
#We create a loop to create the final dataset with all features
temp = df_new[numeric_columns]
for column in categorical_columns:
    temp_df = create_ohe(df_new,column)
    temp = pd.concat([temp,temp_df],axis=1)
print("Shape of Data:",temp.shape)
print("Distinct Datatypes:",temp.dtypes.unique())
print(temp.columns[temp.dtypes=="object"])
temp["StateHoliday"].unique()