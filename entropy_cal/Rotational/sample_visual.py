# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:34:06 2019

pce test with fix training data
@author: huijie
"""


# Work on ethylene dataset

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract, printWaveOut
from DA_PES.utils import parity, error_cal, sub_hist
from DA_PES.vibfreq import read_hess, vibcal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#np.random.seed(10)
nd, fulllist = read_data('./output/energy-forces-sampling.txt', 5)

_, origin = read_data('./output/energy-forces-min.txt', 5)

xfull, yfull, _, _ = extract(fulllist, origin[0], shift=True, scale=1)

plt.figure()
plt.scatter(xfull[:, 0], xfull[:,1])


for i in range(144):
    print(xfull[i, :2])


