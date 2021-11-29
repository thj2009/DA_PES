# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:53:42 2019

@author: huijie
"""

# EXTRACT SYMMETRIC

# Work on ethylene dataset

import numpy as np
np.printoptions(threshold=np.inf)
import DA_PES
from DA_PES import D_opt
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal, weight_construct
from DA_PES.vibfreq import read_hess, vibcal
from sklearn.model_selection import train_test_split


nd, datalist = read_data('./output/test_symmetric.txt', 7)
_, origin = read_data('./output/origin.txt', 7)

xfull, yfull, xdev_full, dydx_full = extract(datalist, origin[0], shift=True)

H = np.zeros([21, 21])
for i in range(42):
    _x = xfull[i, :]
#    print(_x.tolist().index(0.015))
#    print(_x.tolist())
    pos = np.argwhere(_x > 0)
    neg = np.argwhere(_x < 0)
    print(pos, neg)
H = np.array(dydx_full[:21]) - np.array(dydx_full[21:])

newH = H / (2 * 0.015)

HH = 1/2. * (newH + newH.T)

HH = -newH

fp = open('HNF2.txt', 'w')
for i in range(21):
    for j in range(21):
        fp.write('{0:^12.5f}'.format(HH[i, j]))
    fp.write('\n')
fp.close()


HH = 1/2. * (newH + newH.T)
# Calculate wave number
atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

freq, wave = vibcal(HH, atoms)
