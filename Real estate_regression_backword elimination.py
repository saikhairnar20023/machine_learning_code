# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:13:32 2023

@author: Sai khairnar
"""
import pandas as pd

dataset=pd.read_csv('C:/Users/ABC/Documents/Business_analytics_python/DATA/Real estate.csv')

dataset.drop(["No",],axis=1,inplace=True)
dataset.head(5)

x=dataset.iloc[:,[0,1,2,3,4,5]]
y=dataset.iloc[:,-1]
               
            
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

regressor.coef_

regressor.intercept_

y_pred = regressor.predict(x_test)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

import statsmodels.api as sm

import numpy as nm
x = nm.append(arr = nm.ones((414,1)).astype(int), values=x, axis=1)

x_opt=x[:, [ 0,1,2,3,4,5,6]]

regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()
#-------------------------------------------------
x_opt=x[:, [ 0,1,2,3,4,5]]

regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


data_set= pd.read_csv('C:/Users/ABC/Documents/Business_analytics_python/DATA/Real estate.csv') 
#Extracting Independent and dependent Variable  
x_BE= data_set.iloc[:,[0,1,2,3,4,5]].values
y_BE= data_set.iloc[:,-1].values 
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.2, random_state=0)

#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_BE_test)

#Cheking the score  
#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)
#The above score tells that our model is now more accurate with the test dataset with
#accuracy equal to 65%

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)



















