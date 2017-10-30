# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:59:47 2017

@author: Asanka
"""

import tensorflow as tf

x1=tf.constant(4)
x2=tf.constant(6)
result=tf.multiply(x1,x2)

sess=tf.Session() #start the session -need to run session to do a computation
print(sess.run(result))
sess.close()

with tf.Session() as sess:#alternate session -automatically closes the session
    print(sess.run(result))