# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:54:36 2017

@author: Asanka
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5,6], dtype=np.float64)
y=np.array([5,4,6,5,6,7], dtype=np.float64)

def  bestFitSlopeAndIntercept(x,y):
    m=( mean(x)*mean(y)-mean(x*y) ) / (mean(x)*mean(x) -mean(x*x))
    b=mean(y)-m*mean(x)
    return m,b

m,b=bestFitSlopeAndIntercept(x,y)

regressoionLine=[(m*xVal)+b  for xVal in x]

plt.scatter(x,y)
plt.plot(x,regressionLine)
plt.show()