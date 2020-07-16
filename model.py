#! /usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Spiliting the data
Xtrain,Xtest,yTrain,yTest = train_test_split(X,y,test_size = .3,random_state = 0)

regr = LinearRegression()
regr.fit(Xtrain,yTrain)



ypred = regr.predict(Xtest)

pickle.dump(regr,open('model.pkl','wb'))

#loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1.8]]))


