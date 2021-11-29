#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 22:52:36 2019

@author: thj2009
"""

# testing second order polynomial expansion model
#import pytest 
import DA_PES
from DA_PES.utils import parity
import numpy as np

# order of polynomial 
p = 2
# number of independent variable 
n = 7

np.random.seed(1000)
#x = np.random.normal(0, 1, 100); x = np.expand_dims(x, axis=1)
#xtest = np.random.normal(0, 1, 100); xtest = np.expand_dims(xtest, axis=1)

x = np.random.uniform(-1, 1, 100); x = np.expand_dims(x, axis=1)


#pce = DA_PES.PCE(1, 15, withdev=True, copy_xy=True, polytype='Legd')
#pce = DA_PES.PCE(1, 15, withdev=True, copy_xy=True, polytype='Herm')
pce = DA_PES.PCE(1, 15, withdev=True, copy_xy=True, polytype='Plain')


# =============================================================================
# TRAIN
# =============================================================================
y = np.sin(x*2)
dy2dx = -4 * np.sin(x*2)
dydx = 2 * np.cos(x*2)


# =============================================================================
# TEST
# =============================================================================
ytest = np.sin(xtest*2)
dy2dx_test = -4 * np.sin(xtest*2)
dydx_test = 2 * np.cos(xtest*2)

pce.fit(x, y, xdev=x, dydx=dydx)

# =============================================================================
# PREDICTION
# =============================================================================
ypred = pce.predict(x)

ytest_pred = pce.predict(xtest)


#dydx_test_pred = []
#for _x in xtest:
#    dydx_test_pred.append(pce.derivative(m=1, x=_x))



dydx_train_pred = []
for _x in x:
    dydx_train_pred.append(pce.derivative(m=1, x=_x))
    
parity(y, ypred)
#parity(ytest, ytest_pred)
#parity(dydx_test, dydx_test_pred)
parity(dydx, dydx_train_pred)

#dev = []
#for idx, _x in enumerate(x):
#    dy = pce.derivative2(m=1, x=_x)
#    dev.append(dy)
#    print(dydx[idx], dy)
#
#
parity(pce.Y, pce.X.dot(pce.coef))
#
#dev = []
#for idx, _x in enumerate(x):
#    dy = pce.derivative2(m=2, x=_x)
#    dev.append(dy)
#    print(dy2dx[idx], dy)
#
#
#parity(dy2dx, dev)

#yy = pce.X.dot(pce.coef)
#parity(pce.Y, yy)