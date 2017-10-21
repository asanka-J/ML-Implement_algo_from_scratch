# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 07:36:43 2017

@author: Asanka
"""
#Dataset :  stock prices, dividends and splits for 3000 US publicly-traded companies #

import numpy as np
import pandas as pd
import quandl as Quandl
import math
from sklearn import preprocessing, svm
from sklearn import model_selection as cross_validation
from sklearn.linear_model  import LinearRegression
 
df=Quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0 #High level percent
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0 #PercentChange for the day

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

pred_adjust_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

pred_adjust_out = int(math.ceil(0.01*len(df)))

df['label'] = df[pred_adjust_col].shift(-pred_adjust_out)
df.dropna(inplace=True)


X=np.array(df.drop(['label'],1))
Y=np.array(df['label'])

X=preprocessing.scale(X)
#Y=preprocessing.scale(Y)

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

clf=LinearRegression()
clf.fit(X_train,Y_train)
accuracy=clf.score(X_test,Y_test)
print(df.head())

  
 