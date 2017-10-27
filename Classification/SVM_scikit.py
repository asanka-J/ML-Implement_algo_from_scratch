# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 07:26:15 2017

@author: Asanka

dataSet : breast cancer databases was obtained from the University of Wisconsin
          Hospitals, Madison from Dr. William H. Wolberg. 
""""

import numpy as np
import pandas as pd
from sklearn import preprocessing ,svm
from sklearn import model_selection as cross_validation

df=pd.read_csv('breast-cancer-wisconsin.data.csv')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
clf=svm.SVC()
clf.fit(X_train,Y_train)

accuracy=clf.score(X_test,Y_test)
print('Accuracy : ',accuracy)

testInput=np.array([[3,4,4,5,7,10,3,2,1],
                    [1,1,2,4,7,2,3,2,1]])
testInput=testInput.reshape(len(testInput), -1)
prediction = clf.predict(testInput)
print('Prediction : ', prediction)

