# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:20:36 2023

@author: Sai Khairnar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

liver_data = pd.read_csv('C:/Users/ABC/Documents/Business_analytics_python/DATA/indian_liver_patient.csv') 
print(liver_data)

sns.countplot(x="Gender",data=liver_data)

liver_data['Age'].plot.hist()

liver_data.info()
pd.get_dummies(liver_data,["Gender"],drop_first=(True))
liver_data_dummy =pd.get_dummies(liver_data,["Gender"],drop_first=(True))
liver_data_dummy.head(5)

liver_data = liver_data_dummy
liver_data.head(5)

liver_data.drop(["Gender"],axis=1,inplace=True)
liver_data

liver_data.isnull()
liver_data.isnull().sum()

liver_data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
sns.heatmap(liver_data.isnull(),yticklabels=False)



x = liver_data.iloc[:,[0,1,2,3,4,5,6,7,8,10]].values
y = liver_data.iloc[:,9].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)

logmodel.coef_

logmodel.intercept_

y_pred = logmodel.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)


liver_data1=liver_data
liver_data
#----------------------------------------------------------------------------

import statsmodels.api as sm
import numpy as nm

x1=liver_data1.iloc[:,[0,1,2,3,4,5,6,7,8,10]].values
y1=liver_data1.iloc[:,9].values


x1 = nm.append(arr = nm.ones((583,1)).astype(int), values=x1, axis=1)

x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10]]

regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()

x_opt= x1[:, [0,1,3,4,5,6,7,8,10]]

regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


x_opt= x1[:, [0,1,3,4,5,7,8,10]]

regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()

x_opt= x1[:, [0,1,3,4,5,7,8]]

regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.2, random_state=2)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_BE_test,predictions)
accuracy_score(y_BE_test,predictions)























