# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 04:39:12 2017

@author: Asanka
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from collections import counter

plot1=[1,3] 
plot2=[2,4]

def Euclidean_distance():
    E_distance=sqrt( (plot1[0]-plot2[0])**2 +(plot1[1]-plot2[1])**2 )
    return E_distance
 
print(Euclidean_distance())


dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,3]] }
newInput=[5,7]

[
 print(dataset[ii]) for ii in dataset
 ]