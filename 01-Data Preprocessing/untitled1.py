#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:35:23 2017

@author: alexz
"""

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import os



#importing dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y =  dataset.iloc[:,3].values




#handling missing data

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", 
                  axis = 0, verbose = 0)

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])


    
#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder_X = LabelEncoder()     
X[:,0] = labelEncoder_X.fit_transform(X[:,0])


oneHotEncoder  = OneHotEncoder(categorical_features = [0], dtype=np.int)
X = oneHotEncoder.fit_transform(X).toarray()


for row in X:
    for c in row:
        print('{0}'.format(c, ), end= ' ')
    print()

labelEncoder_Y = LabelEncoder()     
y = labelEncoder_Y.fit_transform(y)



#splits to learning and test sets

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2,
                                                    random_state = 0)


