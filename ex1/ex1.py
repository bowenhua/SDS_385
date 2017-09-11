#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:03:26 2017

@author: bowenMac
"""

import numpy as np
import scipy as sp
from time import time

def initialize(n, p):
    X = np.random.rand(n,p)
    W_vector = np.random.rand(n,1) #If we want to diagnolize it we cannot define W as (p,1) matrix
    y = np.random.rand(n,1)
    return (X,y, W_vector)

def solve_linear1(X,y,W_vector): # Naive method
    WX = X*W_vector # Broadcasting
    C = (X.T).dot(WX)
    C = np.linalg.inv(C)
    return C.dot(X.T).dot(y*W_vector)

def solve_linear2(X,y,W_vector): # Pseudoinverse 
    W_vector = np.sqrt(W_vector)
    X = X * W_vector
    y = y * W_vector
    pinv_X = np.linalg.pinv(X)
    return pinv_X.dot(y)

def solve_linear3(X,y,W_vector): # Cholesky
    C = (X.T).dot(X*W_vector)
    d = (X.T).dot(y*W_vector)
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L,d)
    return np.linalg.solve(L.T,alpha)

def solve_linear4(X,y,W_vector): # Sparse lsqr   
    beta,_,_,_,_,_,_,_,_,_ = sp.sparse.linalg.lsqr(X, y)
    return beta
    
    

## Problem (C)
np_comb = [(2000, 50), (1000,1000), (20000,50), (50000,50), (5000,5000)]   
time1 = []
time2 = []
time3 = []
beta1 = []
beta2 = []
beta3 = []
for i in range(len(np_comb)):
    (n,p) = np_comb[i]
    (X,y,W_vector) = initialize(n,p)
    
    start = time()
    beta1.append(solve_linear1(X,y,W_vector))
    end = time()
    time1.append(end-start)
    
    start = time()
    beta2.append(solve_linear2(X,y,W_vector))
    end = time()
    time2.append(end-start)
    
    start = time()
    beta3.append(solve_linear3(X,y,W_vector))
    end = time()
    time3.append(end-start)
  
### Problem (D)
density = [0.01,0.1,0.25,0.5, 1]
(n,p) = (200000,50)

time1_sparse  = []
time2_sparse  = []
time3_sparse  = []
time4_sparse  = []
beta1_sparse = []
beta2_sparse = []
beta3_sparse = []
beta4_sparse = []

for dens in density:
    X = sp.sparse.random(n, p, density=dens)
    X_dense = X.toarray()
    W_vector = np.ones((n,1)) # Assume ones
    y = np.random.rand(n,1)
    
    start = time()
    beta4_sparse.append(solve_linear4(X,y,W_vector))
    end = time()
    time4_sparse.append(end-start)
    
    start = time()
    beta1_sparse.append(solve_linear1(X_dense,y,W_vector))
    end = time()
    time1_sparse.append(end-start)
    
    start = time()
    beta2_sparse.append(solve_linear2(X_dense,y,W_vector))
    end = time()
    time2_sparse.append(end-start)
    
    start = time()
    beta3_sparse.append(solve_linear3(X_dense,y,W_vector))
    end = time()
    time3_sparse.append(end-start)
    
    
    


    
