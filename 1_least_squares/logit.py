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
    """

    s = 1/(1+ np.exp(-z))
    
    return s

def hessian(beta, X):
    """
    Compute the Hessian X^TWX

    """
    w = sigmoid(np.dot(X, beta))
    w_vector = w * (1-w)
    
    return np.dot(X.T, X*w_vector)

def propagate(beta, X, y):
    """

    Return:
    cost -- negative log-likelihood cost for logistic regression (without the constant term)
    dw -- gradient of the loss with respect to w, thus same shape as w
    """
    
    w = sigmoid(np.dot(X, beta))                                     # compute activation
    cost = -(np.dot(y.T, np.log(w)) + np.dot((1 - y).T, np.log(1-w)))                          # compute cost
  
    dbeta = np.dot(X.T, (w-y))
    
    return dbeta, cost


def optimize(beta, X, y, num_iterations, step_size):
    """
    This function optimizes beta by running a gradient descent algorithm
    """
    
    costs = []
    #variable step size
    if step_size == '1/k':
        for i in range(num_iterations):      
            dbeta, cost = propagate(beta, X, y) 
            if max(abs(dbeta))<1e-7:
                break
            beta -= dbeta * (1/(num_iterations))  
            costs.append(cost.flatten())
    elif step_size == 'newton':
        for i in range(num_iterations):
            dbeta, cost = propagate(beta, X, y)
            if max(abs(dbeta))<1e-7:
                break
            delta_beta = np.linalg.solve(hessian(beta, X),dbeta)            
            beta -= delta_beta  
            costs.append(cost.flatten())
    else: # constant step size
        step_size = float(step_size)
        for i in range(num_iterations):
            dbeta, cost = propagate(beta, X, y)  
            if max(abs(dbeta))<1e-7:
                break
            beta -= dbeta * step_size  
            costs.append(cost.flatten())
    
    
    return beta, costs

def predict(beta, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters 
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
traindf[1] = traindf[1].map({"M":1, "B": 0})

trainY = traindf[1].values
trainY = trainY.reshape(trainY.shape[0],1)
trainX = traindf.drop(1, 1)
trainX = trainX.values

# Scaling
scaler = preprocessing.StandardScaler().fit(trainX)
trainX = scaler.transform(trainX)

# Add a column of ones after scaling. If added before scaling this new column
# would have been a column of zeros.
trainX = np.insert(trainX, 10, 1, axis=1) 

#%% Train
beta = np.random.rand(11,1) 
beta1 = beta.copy()
beta2 = beta.copy()
beta3 = beta.copy()
beta4 = beta.copy()
beta5 = beta.copy()

beta1, costs1 = optimize(beta1, trainX, trainY, 50, '0.001')
beta2, costs2 = optimize(beta2, trainX, trainY, 50, '0.005')
beta3, costs3 = optimize(beta3, trainX, trainY, 50, '0.01')
beta4, costs4 = optimize(beta4, trainX, trainY, 50, '1/k')
beta5, costs5 = optimize(beta5, trainX, trainY, 50, 'newton')
#%% Plot

plt.figure()
plt.plot(costs1, label = '0.001')
plt.plot(costs2, label = '0.005')
plt.plot(costs3, label = '0.01')
plt.plot(costs4, label = '1/k')
plt.plot(costs5, label = 'newton')

plt.ylabel('Loss')
plt.xlabel('Iteration Count')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('fig_logit.pdf', format = 'pdf')





#%% Predict
Y_prediction = predict(beta3, trainX)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - trainY)) * 100))