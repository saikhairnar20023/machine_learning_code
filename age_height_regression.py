# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 13:31:02 2023

@author: Sai khairnar
"""

# Simple Linear Regression

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/HP/Documents/skilledge python programs/data/age_height.csv')
print(type(dataset))

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Note: The parameter 'random_state' is used to randomly bifurcate the dataset into training &
#testing datasets. That number should be supplied as arguments to parameter 'random_state'
#which helps us get the max accuracy. And that number is decided by hit & trial method.
    
# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(regressor.coef_)

print(regressor.intercept_)

y_pred=regressor.predict(X_test)


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


import matplotlib.pyplot as plt

plt.plot(X_test, y_test,'--',c='red',)
plt.plot(X_test, y_pred,':',c='blue')











