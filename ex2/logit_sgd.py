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
import random

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


def optimize_sgd(beta, X, y, num_iterations, step_size):
    """
    This function optimizes beta by running SGD
    """
    
    N = X.shape[0]
    P = X.shape[1]
    costs = []
    #variable step size
    if step_size == 'rm': #Robbins–Monro rule
        t0 = 2
        C = 1
        alpha = 0.5
        for i in range(num_iterations):    
            j = random.randint(0,N-1) #Randomly sample a datapoint with replacement
            
            # Here I only pick a slice of X and an entry of y
            # To reuse our codes for standard GD, 
            dbeta, cost = propagate(beta, X[j,:].reshape(1,P), y[j,:].reshape(1,1))  
            
            beta -= dbeta * C * ((num_iterations + t0)**(-alpha))  
            
            if i%1000 == 0:
                _, cost = propagate(beta, X, y) 
                costs.append(cost.flatten())
            
    else: # constant step size
        step_size = float(step_size)
        for i in range(num_iterations):
            j = random.randint(0,N-1) #Randomly sample a datapoint with replacement
            
            # Here I only pick a slice of X and an entry of y
            # To reuse our codes for standard GD, 
            dbeta, cost = propagate(beta, X[j,:].reshape(1,P), y[j,:].reshape(1,1))  
            
            beta -= dbeta * step_size
            
            if i%1000 == 0:
                _, cost = propagate(beta, X, y) 
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

beta, costs = optimize_sgd(beta, trainX, trainY, 50000, '0.0001')
beta2, costs2 = optimize_sgd(beta2, trainX, trainY, 50000, '0.001')
beta3, costs3 = optimize_sgd(beta3, trainX, trainY, 50000, '0.01')
beta1, costs1 = optimize_sgd(beta1, trainX, trainY, 50000, 'rm')
#%% Plot

plt.figure()
plt.plot(costs, label = '0.0001')
plt.plot(costs1, label = 'rm')
plt.plot(costs2, label = '0.001')
plt.plot(costs3, label = '0.01')

plt.ylabel('Loss')
plt.xlabel(r'Iteration Count ($10^3$)')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('fig_logit.pdf', format = 'pdf')





#%% Predict
Y_prediction = predict(beta1, trainX)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - trainY)) * 100))