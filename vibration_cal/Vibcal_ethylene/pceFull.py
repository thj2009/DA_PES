# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:15:40 2019

@author: huijie
"""

# test different scale


# Work on ethylene dataset

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal, weight_construct
from DA_PES.vibfreq import read_hess, vibcal
from sklearn.model_selection import train_test_split
# =============================================================================
# SETTING
# =============================================================================

#seed = np.random.randint(1000)
seed = 10000
#scl = 0.045
scl = 0.015
#scl = 0.010

singlepts = './output/test_%.3f_chris.txt' %scl


regr = 'OLS'
order = 2
scale = 1
DOF = 21
repeat = 1
polytype = 'Herm'

atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

np.random.seed(seed)
nd, datalist = read_data(singlepts, 7)
_, origin = read_data('./output/origin.txt', 7)


H_ib7 = read_hess('./output/hessian/IB7_AUTO.txt', sym=True)
H_ib5nf1 = read_hess('./output/hessian/IB5NF1.txt', sym=True)
H_ib5nf2 = read_hess('./output/hessian/IB5NF2.txt', sym=True)
H_ib5nf4 = read_hess('./output/hessian/IB5NF4.txt', sym=True)


freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)
freq_nf1, wave_nf1 = vibcal(H_ib5nf1, atoms)
freq_nf2, wave_nf2 = vibcal(H_ib5nf2, atoms)
freq_nf4, wave_nf4 = vibcal(H_ib5nf4, atoms)

#train_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
#train_sizes = [27, 36, 45, 54, 63, 72, 81, 90, 99, 100]
train_sizes = [100]

rmse, std = [], []
pce = DA_PES.PCE(21, order, withdev=True, copy_xy=True, regr=regr, polytype=polytype)

#data_train, data_test = train_test_split(datalist, test_size=0)

data_train = datalist
xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)

weights = 1
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)
Hpce = pce.derivative(m=2) / (scale**2)

freq_pce, wave_pce = vibcal(Hpce, atoms)


nsample = 50000
wave_list = []
Hlist = pce.uq_2nd_order(nsample=nsample)
for _H in Hlist:
    _freq, _wave = vibcal(_H / (scale**2) , atoms)
    wave_list.append(_wave)

wave_mean = np.mean(wave_list, axis=0)
wave_std = np.std(wave_list, axis=0)


data = {}
data['wave_pce'] = wave_pce.tolist()
data['wave_mean'] = wave_mean.tolist()
data['wave_std'] = wave_std.tolist()
data['nsample'] = nsample

data['ib7'] = wave_ib7.tolist()
data['ib5nf4'] = wave_nf4.tolist()
data['ib5nf2'] = wave_nf2.tolist()
data['ib5nf1'] = wave_nf1.tolist()

# Save to json
import json
with open('./result/result_%.3f_full.txt' %scl, 'w') as fp:
    json.dump(data, fp, indent=4)

