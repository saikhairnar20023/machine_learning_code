# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:28:37 2023

@author: Sai khairnar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Social_data=pd.read_csv("C:/Users/ABC/Documents/Business_analytics_python/DATA/Social_Network_Ads.csv")
print(Social_data)

sns.countplot(x="Purchased",data=Social_data)

Social_data["Age"].plot.hist()

Social_data["EstimatedSalary"].plot.hist()

#get dummy value of gender
pd.get_dummies(Social_data['Gender'],drop_first=(True))
gender_dummy = pd.get_dummies(Social_data['Gender'],drop_first=(True))
gender_dummy.head(5)

Social_data = pd.concat([Social_data,gender_dummy],axis=1)
Social_data


Social_data.drop(["Gender","User ID"],axis=1,inplace=True)
Social_data.head(5)

x = Social_data.iloc[:,[0,1,3]].values
y = Social_data.iloc[:,2].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

logmodel.coef_

logmodel.intercept_

y_pred = logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)



Social_data1= Social_data
Social_data1.head(5)



#-----------------------------------------------------------------
#backword elimination
import statsmodels.api as sm
import numpy as nm


x1=Social_data1.iloc[:,[0,1,3]].values
y1=Social_data.iloc[:,2].values

x1 = nm.append(arr = nm.ones((400,1)).astype(int), values=x1, axis=1)

#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt= x1[:, [0,1,3]]

#for fitting the model, we will create a regressor_OLS object of new class OLS of statsmodels library. 
#Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


x_opt= x1[:, [0,1]]
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()
regressor_OLS.summary()

from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.25, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_BE_test,predictions)
accuracy_score(y_BE_test,predictions)













