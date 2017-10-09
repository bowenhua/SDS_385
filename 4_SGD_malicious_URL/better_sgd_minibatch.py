#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 09:58:46 2017

@author: bowenhua
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from numba import jit
from time import time
from sklearn.model_selection import train_test_split

#%% Functions

def sigmoid(z):
    """
    Compute the sigmoid of z
    """

    s = 1/(1+ np.exp(-z))
    
    return s

def random_mini_batches(X, Y, mini_batch_size = 64 ):
    """
    Creates a list of random minibatches from (X, Y)
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    N = X.shape[0]                  
    mini_batches = []
        
    permutation = list(np.random.permutation(N))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = N//mini_batch_size # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size] # We do not have to specify columns for scipy.sparse
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size, :]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if N % mini_batch_size != 0:

        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size:] # We do not have to specify columns for scipy.sparse
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size:,:]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def propagate_lasso(beta, X, y, lam = 0.1, compute_cost = False):
    """
    lam is lambda for the l-1 penalty term
    
    Return:
    cost -- negative log-likelihood cost for logistic regression (without the constant term)
    dbeta -- gradient of the loss with respect to beta
    """
    
    w = sigmoid(X * beta) # Just use * for sparse matrix multiplication
    
    if compute_cost:
        cost = -(np.dot(y.T, np.log(w + 1e-7)) + np.dot((1 - y).T, np.log(1-w + 1e-7)))/X.shape[0] +  lam * np.linalg.norm(beta, 1)                  
        return cost
    else:
        dbeta = X.T * (w-y)/X.shape[0] + lam * (beta > 0).astype(float) - lam * (beta < 0).astype(float)
        return dbeta
    
    

def optimize_adagrad(beta, minibatches, num_epochs, eta= 0.01, lam = 0.01):
    """
    eta is the step size in AdaGrad
    
    This function optimizes beta by running AdaGrad
    """
    epsilon = 1e-7 #smoothing factor
    P = X.shape[1]
    sos_grad = np.zeros((P,1))
    costs = []
    N_batch = len(minibatches)
 
    for i in range(num_epochs):
        
        for j in np.random.permutation(N_batch):
            
            dbeta = propagate_lasso(beta, minibatches[j][0], minibatches[j][1], lam, compute_cost = False)   
            
            sos_grad += np.square(dbeta)
            
            beta -= (1/np.sqrt(sos_grad + epsilon)) * dbeta * eta
        

        if i%10 == 0:
            cost = propagate_lasso(beta, X, y, lam, compute_cost = True) 
            costs.append(cost.flatten())
            
            if len(costs) >= 2 and costs[-2] - costs[-1] < 1e-6:
                break
    
    
    return beta, costs

def predict(beta, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters 
    '''
    m = X.shape[0]
    Y_prediction = np.zeros((m,1))

    w = sigmoid(X * beta)    
    
    for i in range(w.shape[0]):
        if w[i,0] > 0.5:
            Y_prediction[i,0] = 1
        else:
            Y_prediction[i,0] = 0
    
    return Y_prediction
#%% Read data


X = sp.sparse.load_npz('x_all.npz')
y = np.load('y_all.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

minibatches = random_mini_batches(X_train,y_train, mini_batch_size = 65536)

#%% Train

beta = np.random.rand(X_train.shape[1],1) 

start = time()
# Did some manual CV
beta, costs = optimize_adagrad(beta, minibatches, 300, eta = 1, lam = 1e-6)
end = time()

print(end-start)
#%% Plot

plt.figure()

plt.plot(costs, label = 'AdaGrad')

plt.ylabel('Loss')
plt.xlabel(r'Epoch Count ($\times 10$)')
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('fig_logit.pdf', format = 'pdf')





#%% Predict
Y_prediction_train = predict(beta, X_train)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))

Y_prediction_test = predict(beta, X_test)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))