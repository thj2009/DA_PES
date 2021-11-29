# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 15:22:05 2019

@author: huijie
"""

# test mappling function

# testing second order polynomial expansion model
import DA_PES
from DA_PES.utils import parity
import numpy as np

# order of polynomial 
p = 2
# number of independent variable 
n = 7

pce = DA_PES.PCE(2, 2, withdev=False, copy_xy=True)

x = np.random.normal(0, 1, (1000, 2))

def testfunc(x, m=0):
    if m == 0:
        return np.sin(x[0]) * x[1]**2
    elif m == 1:
        return np.array([np.cos(x[1]) * x[1]**2, np.sin(x[0] * 2 * x[1])])
    elif m == 2:
        return np.array([[-np.sin(x[0]) * x[1]**2, 2 * np.cos(x[0])*x[1]],
                          [2*np.cos(x[0])*x[1], 2 * np.sin(x[0])]])

y = np.array([testfunc(_x) for _x in x])
dy2dx = [testfunc(_x, m=2) for _x in x]

pce.fit(x, y, uncer=True)

pce.map_to_plain()

print(pce.derivative(m=2))
print(pce.derivative2(m=2))

hess_list = pce.uq_2nd_order(nsample=1000)
HH = np.array(hess_list)

mean_HH = np.mean(HH, axis=0)
