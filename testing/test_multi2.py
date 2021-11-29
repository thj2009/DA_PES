# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:59:37 2019

creat fictious hessian 

to test higher order derivative

@author: huijie
"""

import numpy as np


#np.random.seed(1000)
nvar=2


def testfunc(x, m=0):
    a = 0.5; b = 0.1; c = 0.1; d = 0.2
    if m == 0:
        y = a*x[ 0]**2 + b*x[ 1]**3 + c*x[0]*x[1] + d*x[0]**2*x[1]
    if m == 1:
        y = [2*a*x[0]+c*x[1]+2*d*x[ 0]*x[1],
             3*b*x[1]**2+c*x[0]+d*x[0]**2]
    if m == 2:
        y =[[2*a+2*d*x[1], c+2*d*x[0]],
            [c+2*d*x[0], 3*b*2*x[1]]]
    return y

xtrain = np.random.uniform(-1, 1, (3, nvar))
xtest = np.random.uniform(-1, 1, (10, nvar))

ytrain = [testfunc(_x) for _x in xtrain]
dydx_train = [testfunc(_x, 1) for _x in xtrain]
ytest = [testfunc(_x) for _x in xtest]
dydx_test = [testfunc(_x, 1) for _x in xtest]


import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal
from DA_PES.vibfreq import read_hess, vibcal


pce = DA_PES.PCE(nvar, 4, withdev=True, copy_xy=True, regr='OLS')

pce.fit(xtrain, ytrain, xdev=xtrain, dydx=dydx_train, uncer=True, njob=2)



#dydx_pred = [pce.derivative(m=1, x=_x) for _x in xtest]
#ytest_pred = pce.predict(xtest)

#parity(ytest, ytest_pred)
#parity(dydx_test, dydx_pred)


H = testfunc([0, 0], m=2)
Hpred = pce.derivative(m=2)


parity(H, Hpred)




