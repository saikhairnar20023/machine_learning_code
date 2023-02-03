# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:30:57 2023

@author: Sai khairnar
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_data=pd.read_csv('C:/Users/ABC/Documents/Business_analytics_python/DATA/iris (1).csv')
print(iris_data)

iris_data.info()

#Handling missing value
iris_data.isnull()
iris_data.isnull().sum()


iris_data.drop(["Id"],axis=1,inplace=True)
iris_data.head(5)

x=iris_data.iloc[:,[0,1,2,3]].values
y=iris_data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)

y_pred=logmodel.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion_matrix(y_test, y_pred)
accuracy_score(y_test,y_pred)








