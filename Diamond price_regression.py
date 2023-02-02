# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 13:47:21 2023

@author: Sai khairnar
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv("C:/Users/ABC/Documents/Business_analytics_python/DATA/diamonds.csv")
print(dataset)


pd.get_dummies(dataset["cut"])
pd.get_dummies(dataset["cut"],drop_first=True)
S_Dummy = pd.get_dummies(dataset["cut"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.
dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["cut",],axis=1,inplace=True)
dataset.head(5)
#---------------------

pd.get_dummies(dataset["color"])
pd.get_dummies(dataset["color"],drop_first=True)
S_Dummy = pd.get_dummies(dataset["color"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.
dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["color",],axis=1,inplace=True)
dataset.head(5)
#---------------------
pd.get_dummies(dataset["clarity"])
pd.get_dummies(dataset["clarity"],drop_first=True)
S_Dummy = pd.get_dummies(dataset["clarity"],drop_first=True)
S_Dummy.head(5)
#Now, lets concatenate these dummy var columns in our dataset.
dataset = pd.concat([dataset,S_Dummy],axis=1)
dataset.head(5)
#dropping the columns whose dummy var have been created
dataset.drop(["clarity",],axis=1,inplace=True)
dataset.head(5)
#---------------------

dataset.drop(["x",],axis=1,inplace=True)
dataset.head(5)

dataset.drop(["y",],axis=1,inplace=True)
dataset.head(5)

dataset.drop(["z",],axis=1,inplace=True)
dataset.head(5)

X = dataset.drop("price",axis=1)
y = dataset["price"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

regressor.coef_

regressor.intercept_


y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

#-------------------------------------------------------------------

#backword elimination
import statsmodels.api as sm
import numpy as np


X_1 = dataset.drop("price",axis=1)
y_1 = dataset["price"]

X = np.append(arr = np.ones((53940,1)).astype(int), values=X, axis=1)

x_opt=X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]]
regressor_OLS=sm.OLS(endog = y_1, exog=x_opt).fit()


regressor_OLS.summary()

















