# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:59:37 2019

creat fictious hessian 

to test higher order derivative

@author: huijie
"""

import numpy as np


#np.random.seed(1000)

nvar = 5
nsample = 100

Hess = np.random.rand(nvar,nvar)
Hess = 1/2. * (Hess + Hess.T)
eig = np.linalg.eigvals(Hess)

while np.any(eig < 0):
    Hess += np.eye(nvar)
    eig = np.linalg.eigvals(Hess)

xtrain = np.random.uniform(-1, 1, (nsample, nvar))
# add noise 
#ytrain = [_x.dot(Hess).dot(_x) + np.random.normal(0,0.5) for _x in xtrain]
#dydx_train = [2 * Hess.dot(_x) + np.random.normal(0,0.5,nvar) for _x in xtrain]
ytrain = [_x.dot(Hess).dot(_x) for _x in xtrain]
dydx_train = [2 * Hess.dot(_x) for _x in xtrain]


xtest = np.random.uniform(-1, 1, (100, nvar))
# add noise 
ytest = [_x.dot(Hess).dot(_x)  for _x in xtest]
dydx_test = [2 * Hess.dot(_x) for _x in xtest]



import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal
from DA_PES.vibfreq import read_hess, vibcal






pce = DA_PES.PCE(nvar, 4, withdev=True, copy_xy=True, regr='LassoCV')
pce.fit(xtrain, ytrain, xdev=xtrain, dydx=dydx_train, uncer=True, njob=2)



dydx_pred = [pce.derivative(m=1, x=_x) for _x in xtest]
ytest_pred = pce.predict(xtest)

parity(ytest, ytest_pred)
parity(dydx_test, dydx_pred)

H = pce.derivative(m=2)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(pce.coef)

parity(Hess * 2, H)

