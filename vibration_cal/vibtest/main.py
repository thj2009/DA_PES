#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 03:06:53 2019

@author: thj2009
"""

# Work on ethylene dataset

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity
from hess_cal import read_hess, vibcal

nd, datalist = read_data('./data/ethylene/perturb_train.txt', 7)
nd_test, datalist_test = read_data('./data/ethylene/perturb.txt', 7)
datalist = datalist[1:]
_, origin = read_data('./data/ethylene/originE.txt', 7)

##datalist
ntrain = 10
index = np.arange(nd-1)
np.random.shuffle(index)

scale = 1
datalist = [datalist[index[i]] for i in range(ntrain)]
x, y, xdev, dydx = extract(datalist, origin[0], shift=True, scale=scale)
xtest, ytest, _, _ = extract(datalist_test, origin[0], shift=True, scale=scale)


#import DA_PES
pce = DA_PES.PCE(21, 3, withdev=True, copy_xy=True, regr='Lasso')
pce.fit(x, y, xdev=xdev, dydx=dydx)

ypred = pce.predict(x)

ypred_test = pce.predict(xtest)

##
parity(y, ypred)

parity(ytest, ypred_test)
H = pce.derivative(m=2) / (scale**2)

H_VASP = read_hess('./data/ethylene/hessian.txt')


mass = [12.0107, 12.0107, 1.00794, 1.00794, 1.00794, 1.00794, 1.00794]
freq, wave = vibcal(H, mass)
freq_vasp, wave_vasp = vibcal(H_VASP, mass)
#
parity(wave_vasp, wave)

