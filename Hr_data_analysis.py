# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:05:21 2024

@author: This Pc
"""
#importing libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#importing dataset
HR_data = pd.read_csv(r"E:\Data Scientist - Course\Project\Python_Project_ML/HR_comma_sep.csv")
HR_data

#Data cleaning and Preparation
HR_data.shape  # [14999 rows * 10 columns]
HR_data.columns
HR_data.isnull().sum() #No null value)

HR_data.head()
HR_data.describe()
HR_data.info()

#Exploratory Data Analysis

plt.figure(figsize=(10,5))
HR_data.groupby(['Department'])['left'].sum().plot.bar()
plt.ylabel('No of emp left')
plt.title('Empl retention in Department',fontsize=16)
plt.show()

#from the barplot we can see that "sales" department has minimum
#retention

plt.figure(figsize=(10,5))
HR_data.groupby(['Department'])['promotion_last_5years'].sum().plot.bar()
plt.ylabel('No of emp left')
plt.title('Promotions in Department',fontsize=18)
plt.show()

#This indicates though maximum promotion are in sales department we see lowest retention. Highest 
#attrition is from sales department only. 

plt.figure(figsize=(10,5))
HR_data.groupby(['satisfaction_level'])['left'].sum().plot.bar()
plt.ylabel('No of emp left',fontsize = 16)
plt.xlabel('Satisfaction level',fontsize=16)
plt.title('satisfaction level vs employee left',fontsize=18)
plt.show()

#most of the employees who left the job are not satisfies and the level is below 50%
#At the same time there are employees who left the job having higher statisfication level 

plt.figure(figsize=(10,5))
HR_data.groupby(['number_project'])['left'].sum().plot.bar()
plt.ylabel('No of emp left',fontsize = 16)
plt.xlabel('No of projects',fontsize=16)
plt.title('No of projects vs employee left',fontsize=18)
plt.show()

#Most of the employees who left have done 2 or less projects


plt.figure(figsize=(10,5))
HR_data.groupby(['Work_accident'])['left'].sum().plot.bar()
plt.ylabel('No of emp left',fontsize = 16)
plt.xlabel('Work_accident',fontsize=16)
plt.title('Work_accident vs employee left',fontsize=18)
plt.show()

#Work accidents does not impact employees leaving the job

plt.figure(figsize=(10,5))
HR_data.groupby(['promotion_last_5years'])['left'].sum().plot.bar()
plt.ylabel('No of emp left',fontsize = 16)
plt.xlabel('promotions',fontsize=16)
plt.title('promotions vs employee left',fontsize=18)
plt.show()

#Most of the employees who left are with 0 promotions

plt.figure(figsize=(10,5))
HR_data.groupby(['salary'])['left'].sum().plot.bar()
plt.ylabel('No of emp left',fontsize = 16)
plt.xlabel('salary',fontsize=16)
plt.title('salary vs employee left',fontsize=18)
plt.show()

#From the above we can say salary is the major factor for employee
#leaving the job

#We will split the data into Train and Test for analysis
#creating copy of the data set

HR_data2 = HR_data.copy()

#dropping the Work accident as it is not impacting the employee retention

HR_data2 = HR_data2.drop('Work_accident',axis=1)

HR_data2.head()

#Now we will convert the categorical values in to dummpy/indicator variables

HR_dummy = pd.get_dummies(HR_data2)
HR_dummy
HR_dummy.head(3)

#separating dependent and independent features

Y = HR_dummy.pop('left')
Y
X = HR_dummy
X

#Splitting train and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)

#Applying feature scaling on the data for better model prediction 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Applying Logistic Regression because it has probability type output
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
confm = confusion_matrix(Y_test,Y_pred)
print(confm)

#This shows that (2656+237)= 2893 values out of 3750 values are same as 
#the actual values 

#printing accuracy score

accuracy_score(Y_test,Y_pred)
#77%

#Hence we can use this model to predict whether an employee will leave job or not. 
