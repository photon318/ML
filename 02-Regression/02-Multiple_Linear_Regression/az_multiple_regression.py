#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:07:06 2017

@author: alexz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.chdir("/Users/Alexz/Documents/Github/ML/02-Regression/02-Multiple_Linear_Regression")


data = pd.read_csv("50_startups.csv");

X  = data.iloc[:,:-1].values
Y = data.iloc[:,4].values



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy variable trap

X = X[:,1:]


#splits to learning and test sets

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 0.2,
                                                    random_state = 0
                                                    )


#train 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#predict
y_pred = regressor.predict(X_test)

#display some charts 

plt.scatter(X_test[:, 2], regressor.predict(X_test))
plt.scatter(X_train[:, 2], regressor.predict(X_train), c = 'red')

plt.show()

#bulding optimal model with using backward elimimination

import statsmodels.formula.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]] 
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

#train new optimized model

X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, Y, 
                                                    test_size = 0.2,
                                                    random_state = 0
                                                    )
#new optimized regressor

regressor_opt = LinearRegression()
regressor_opt.fit(X_train_opt, y_train_opt)


#predict
y_pred_opt = regressor_opt.predict(X_test_opt)

#Display results
plt.scatter(X_train[:, 2], regressor.predict(X_train), c = 'red')
plt.scatter(X_test[:, 2], regressor.predict(X_test), c = 'blue')
plt.scatter(X_train_opt[:, 1], regressor_opt.predict(X_train_opt), c = 'orange')
plt.scatter(X_test_opt[:, 1], regressor_opt.predict(X_test_opt), c = 'green')
plt.xlabel("X0+R&D")
plt.ylabel("Net Profit")
plt.show()



