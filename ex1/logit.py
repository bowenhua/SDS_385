#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:49:53 2017

@author: bowenMac
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt

#%% Functions
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+ np.exp(-z))
    
    return s

def propagate(beta, X, y):
    """

    Return:
    cost -- negative log-likelihood cost for logistic regression (without the constant term)
    dw -- gradient of the loss with respect to w, thus same shape as w
    """
    
    w = sigmoid(np.dot(X, beta))                                     # compute activation
    cost = -(np.dot(y.T, np.log(w)) + np.dot((1 - y).T, np.log(1-w)))                          # compute cost
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
  
    dbeta = np.dot(X.T, (w-y))
    
    return dbeta, cost


def optimize(beta, X, y, num_iterations, learning_rate, print_cost = True):
    """
    This function optimizes w  by running a gradient descent algorithm
    """
    
    costs = []
    
    for i in range(num_iterations):
    
        dbeta, cost = propagate(beta, X, y)
        
        beta -= dbeta * learning_rate

        costs.append(cost.flatten())
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    
    return beta, costs

def predict(beta, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters 
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))

    w = sigmoid(np.dot(X, beta))    
    
    for i in range(w.shape[0]):
        if w[i,0] > 0.5:
            Y_prediction[i,0] = 1
        else:
            Y_prediction[i,0] = 0
    
    return Y_prediction
#%% Read data

traindf = pd.read_csv('wdbc.csv',header=None, usecols=range(1,12))
traindf[12] = np.ones((569,1))
traindf[1] = traindf[1].map({"M":1, "B": 0})

trainY = traindf[1].values
trainY = trainY.reshape(trainY.shape[0],1)
trainX = traindf.drop(1, 1)
trainX = trainX.values

# Scaling
scaler = preprocessing.StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)

#%% Train
beta = np.random.rand(11,1) 
beta, costs = optimize(beta, trainX, trainY, 500, 0.005, print_cost = True)

#%% Plot

plt.figure()
plt.plot(costs)

plt.ylabel('Loss')
plt.xlabel('Iteration Count')
plt.savefig('fig_logit.pdf', format = 'pdf')

#%% Predict
Y_prediction = predict(beta, trainX)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - trainY)) * 100))