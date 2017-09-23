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
