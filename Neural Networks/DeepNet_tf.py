# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 19:20:05 2017

@author: Asanka
"""
#dataset : MNIST is a simple computer vision dataset. It consists of images of handwritten digits

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # one element is kept on and others off

#defining model
n_nodes_hl1=500 #nodes in layer
n_nodes_hl2=500
n_nodes_hl3=500

n_classes=10
batch_size = 100


#height x width
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural_NModel(data):
    #layer=> weights , biases
    hidden_layer1 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden_layer3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_layer1['weights']), hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer2['weights']), hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer3['weights']), hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(input_val):
    prediction = neural_NModel(input_val)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        