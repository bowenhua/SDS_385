#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:04:34 2017

@author: bowenhua
"""

from sklearn.datasets import load_svmlight_file
import scipy as sp
import numpy as np

# Merge data
X,y = load_svmlight_file('Day' + str(0) + '.svm', n_features = 3231962)
for i in range(1, 121):
    X1,y1 = load_svmlight_file('Day' + str(i) + '.svm', n_features = 3231962)
    y = np.concatenate((y,y1), axis = 0)
    X = sp.sparse.vstack([X,X1])

# Add a column of ones
mtx = np.ones((2396130,1))
mtx = sp.sparse.csr_matrix(mtx)
X = sp.sparse.hstack([X, mtx], format = 'csr')

# Convert -1 in y to 0
y = y == 1
y = y.astype(int)
y = y.reshape(2396130,1)

# Save to binary
sp.sparse.save_npz('x_all.npz', X)
np.save('y_all',y)