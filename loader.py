#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: fmerizzi
"""

import numpy as np 
import os
# Util for pwd 
import gzip
cwd = os.getcwd()
# Data needs to be normalized 
from sklearn.preprocessing import MinMaxScaler


    #Produce the unit vector for the network 
def unit_vector(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    #load data 
    test_imgs = np.load("./kmnist-test-imgs.npz")['arr_0']
    test_labels = np.load("./kmnist-test-labels.npz")['arr_0']
    train_imgs = np.load("./train_img_reduced.npz")['arr_0']
    train_labels = np.load("./train_labels_reduced.npz")['arr_0']
    
    #cast 
    test_imgs = test_imgs.astype("float32")
    train_imgs = train_imgs.astype("float32")
    test_labels = test_labels.astype("int64")
    train_labels = train_labels.astype("int64")

    
    # Reshape so that the input is a single array of dimension 784 
    #(rather than 28x28)
    test_imgs = np.reshape(test_imgs,[10000,784])
    train_imgs = np.reshape(train_imgs,[50000,784])
    
    # Data is raw, normalization is necessary to avoid overflow with the sigmoid 
    scaler = MinMaxScaler(feature_range=(0,1))
    test_imgs = scaler.fit_transform(test_imgs)
    train_imgs = scaler.fit_transform(train_imgs)
    
    #print(test_imgs.shape)
    #print(train_imgs.shape)
    #print(test_labels.shape)
    #print(train_labels.shape)
    
    # Transform the data in a zip iterable object 
    training_inputs = [np.reshape(x,[784,1]) for x in train_imgs]
    training_results = [unit_vector(y) for y in train_labels]
    training_data = zip(training_inputs, training_results)
    test_inputs = [np.reshape(x, [784,1]) for x in test_imgs]
    test_data = zip(test_inputs, test_labels)
    
    return (list(training_data), list(test_data))







