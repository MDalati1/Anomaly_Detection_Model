#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 01:47:01 2022

@author: mohamaddalati
"""

# Assignment 6 
import pandas as pd 
import numpy as np 

"""
Task 1. First, import the data and set the column ‘raw_material’ as the target variable and
column ‘time’ as the predictor. Then, split the data into training (70%) and test (30%). 
"""
df = pd.read_csv('/Users/mohamaddalati/Desktop/INSY-662/Assignment6/production.csv')

X = df[['time']]
y = df['raw_material']

# split data into 70% training and 30% test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.30, random_state = 5) 


"""
Task 2. Then, develop a ridge regression model with alpha = 1 using the training data, and test
the performance of the model using the test data. Report the performance of this model below:
"""
from sklearn.linear_model import Ridge 
ridge1 = Ridge(alpha = 1) 
model1 = ridge1.fit(X_train, y_train)
y_test_pred1 = model1.predict(X_test)
# Calculate the MSE 
from sklearn.metrics import mean_squared_error
ridge_penalty_mse = mean_squared_error(y_test, y_test_pred1)
print("RR MSE is:", ridge_penalty_mse )

"""
Task 3: Now, we are going to use an isolation forest model to remove anomalies (i.e., outliers)
in the data so that the predictive performance of the model improves. Develop an isolation
forest model to detect anomalies in this dataset (using both raw_material and time). Use
contamination = 0.1
"""
# Building isolation forest model
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators = 100, contamination = 0.1)
pred = iforest.fit_predict(df)
score = iforest.decision_function(df) # check the anomaly score 
# after locating the anomalies, we want to extract them
# Extracting anomalies
from numpy import where
anomaly_index = where(pred == -1) # defining anomalies 
anomaly_values = df.iloc[anomaly_index] # creating a separate dataset which only has these anomalies 
values = df.iloc[anomaly_index]

"""
Task 4: . Then, remove observations that are flagged as anomalies by the isolation forest model
from the training/test dataset that we split earlier. 
"""
df = df.drop( [7,8,20,32,45,78,80,104,110,112,114,126,127,130,132,144,152,155,182,195] )

X_new = df[['time']]
y_new = df['raw_material']
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_new,y_new, test_size = 0.30, random_state = 5) 

"""
Task 5: Develop a second ridge regression model with alpha=1 using the same training dataset
where anomalies are removed. Test the performance of this new model on the same test dataset
where anomalies are removed. Report the performance of this model below
"""
from sklearn.linear_model import Ridge
ridge2 = Ridge(alpha = 1) 
model2 = ridge2.fit(X_train2, y_train2)
y_test_pred2 = model2.predict(X_test2)
# Calculate the MSE 
from sklearn.metrics import mean_squared_error
ridge_penalty_mse2 = mean_squared_error(y_test2, y_test_pred2)
print("RR MSE is:", ridge_penalty_mse2)
# 0.052076304313701804















