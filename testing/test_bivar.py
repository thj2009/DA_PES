#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:36 2019

@author: thj2009
"""

import sys
sys.path.append('../')
import DA_PES
import numpy as np

# Define polynomial chaos expansion
nvar = 2    # number of variable
nord = 3    # order of polynomial
poly = 'Herm'   # Type of polynomial

pce = DA_PES.PCE(nvar, nord, copy_xy=True, polytype=poly)

# Total term in polynomial expasion
print('Total term = %d' %len(pce.order_list))

# generate train and test dataset
xtrain = np.random.normal(0, 1, (100, 2))
xtest = np.random.normal(0, 1, (100, 2))


# def function and its derivative // y = sin(x0) * x1^2
def testfunc(x, m=0):
    # m: order of derivative
    if m == 0:
        return np.sin(x[0]) * x[1]**2
    elif m == 1:
        return np.array([np.cos(x[0]) * x[1]**2, np.sin(x[0]) * 2 * x[1]])
    elif m == 2:
        return np.array([[-np.sin(x[0]) * x[1]**2, 2 * np.cos(x[0])*x[1]],
                          [2*np.cos(x[0])*x[1], 2 * np.sin(x[0])]])


ytrain = np.array([testfunc(_x) for _x in xtrain])
ytest = np.array([testfunc(_x) for _x in xtest])

# Fit the model
pce.fit(xtrain, ytrain)

# Predict on test dataset
ytest_pred = pce.predict(xtest)

import matplotlib.pyplot as plt
plt.plot(ytest, ytest_pred, 'ro')