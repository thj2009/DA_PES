# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:18:12 2019

@author: huijie
"""

# test on ishigami functinon

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal
from DA_PES.vibfreq import read_hess, vibcal


Xtest = np.random.normal(0, 1, (10000, 3))

def ishigami(X, m=0):
    '''
    function: ishigami
    input: X
           m - order of derivative
    output dY/dX at order m
    '''
    a = 7
    b = 1
    if m == 0:
        Y = np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] * np.sin(X[:, 0])
    if m == 1:
        Y = [np.cos(X[:, 0]) + b * X[:, 2] * np.cos(X[:, 0]),
             2 * a * np.sin(X[:, 1]) * np.cos(X[:, 1]),
             b * np.sin(X[:, 0])]
        Y = np.array(Y).T
    return Y

ytest = ishigami(Xtest)
Ntrains = [500, 1000, 5000, 10000]

pce = DA_PES.PCE(3, 11, withdev=False, copy_xy=False, regr='OLS')

for ntrain in Ntrains:
    Xtrain = np.random.normal(0, 1, (ntrain, 3))
    ytrain = ishigami(Xtrain)
    pce.fit(Xtrain, ytrain)
    ytest_pred = pce.predict(Xtest)
    parity(ytest, ytest_pred)
    rmse, mae = error_cal(ytest, ytest_pred)
    print(rmse)



pce = DA_PES.PCE(3, 11, withdev=True, copy_xy=False, regr='OLS')

for ntrain in Ntrains:
    Xtrain = np.random.normal(0, 1, (ntrain, 3))
    ytrain = ishigami(Xtrain)
    dydx = ishigami(Xtrain, m=1)
    pce.fit(Xtrain, ytrain, Xtrain, dydx)
    ytest_pred = pce.predict(Xtest)
    parity(ytest, ytest_pred)
    rmse, mae = error_cal(ytest, ytest_pred)
    print(rmse)

