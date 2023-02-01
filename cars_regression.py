# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:10:29 2023

@author: Sai Khairnar
"""

# Multiple Linear Regression

# Importing the libraries

'import matplotlib.pyplot as plt'
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/ABC/Documents/Business_analytics_python/DATA/Luxury_cars.csv')

#Method-1 (Handling Categorical Variables)
pd.get_dummies(dataset["Make"])
pd.get_dummies(dataset["Make"],drop_first=(True))
S_Dummy = pd.get_dummies(dataset["Make"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.

dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["Make",],axis=1,inplace=True)
dataset.head(5)

#drop  model
dataset.drop(["Model",],axis=1,inplace=True)
dataset.head(5)

#Method-1 (Handling Categorical Variables)
pd.get_dummies(dataset["Type"])
pd.get_dummies(dataset["Type"],drop_first=(True))
S_Dummy = pd.get_dummies(dataset["Type"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.

dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["Type",],axis=1,inplace=True)
dataset.head(5)


##Method-1 (Handling Categorical Variables)
pd.get_dummies(dataset["Origin"])
pd.get_dummies(dataset["Origin"],drop_first=(True))
S_Dummy = pd.get_dummies(dataset["Origin"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.

dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["Origin",],axis=1,inplace=True)
dataset.head(5)

#Method-1 (Handling Categorical Variables)
pd.get_dummies(dataset["DriveTrain"])
pd.get_dummies(dataset["DriveTrain"],drop_first=(True))
S_Dummy = pd.get_dummies(dataset["DriveTrain"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.

dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["DriveTrain",],axis=1,inplace=True)
dataset.head(5)
#------------------------------------------------------------------------------

#Obtaining DV & IV from the dataset
X = dataset.drop("MPG (Mileage)",axis=1)
y = dataset["MPG (Mileage)"]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Coefficient
regressor.coef_

# Intercept
regressor.intercept_

#The above score tells that our model is 84% accurate with the test dataset.

#--------------------------Backward Elimination--------------------------------
#Backward elimination is a feature selection technique while building a machine learning model. It is used
#to remove those features that do not have significant effect on dependent variable or prediction of output.

#Step: 1- Preparation of Backward Elimination:

#Importing the library:
import statsmodels.api as sm



#Adding a column in matrix of features:
X1 = dataset.drop("MPG (Mileage)",axis=1)
y1 = dataset["MPG (Mileage)"]
    
import numpy as nm
X = nm.append(arr = nm.ones((426,1)).astype(int), values=X, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]]

#for fitting the model, we will create a regressor_OLS object of new class OLS of 
#statsmodels library. Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()



#Now since x5 has highest p-value greater than 0.05, hence, will remove the x1 variable
#(dummy variable) from the table and will refit the model.
x_opt= X[:, [0,1,3,4,5,9,13,17,22,23,28,31,39,42,43,45,46,47,48,49,51,52]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()



#Extracting Independent and dependent Variable  
x_BE = dataset.iloc[:,[0,1,3,4,5,9,13,17,22,23,28,31,39,42,43,45,46,47,48,49,51,52]].values
y_BE = dataset["MPG (Mileage)"]

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
#accuracy equal to 100%

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)