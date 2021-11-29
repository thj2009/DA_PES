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
from DA_PES import D_opt

# =============================================================================
# SETTING
# =============================================================================

seed = np.random.randint(10000)

scl = 0.015
singlepts = './output/test_%.3f_chris.txt' %scl

regr = 'OLS'
order = 2
scale = 1
DOF = 21
polytype = 'Plain'

atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

np.random.seed(seed)
nd, datalist = read_data(singlepts, 7)
_, origin = read_data('./output/origin.txt', 7)
H_ib7 = read_hess('./output/hessian/IB7_AUTO.txt', sym=True)
freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)

#train_sizes = [30, 42, 52, 63, 73, 84]
train_sizes = [21]
#train_sizes = [26, 35, 47, 57, 68, 78]


# SETTING for Dopt
nstart = 50; ninit= 1

xfull, yfull, xdev_full, dydx_full = extract(datalist, origin[0], shift=True, scale=scale)

pce = DA_PES.PCE(21, order, withdev=True, copy_xy=True, regr=regr)
X = pce.expand_x(pce.order_list, xfull, polytype)
Xexp = pce.expand_dev_xy(pce.order_list, xdev_full, polytype)

# %% Experimental design
Xorigin = []
for i in range(100):
    _x = X[i, :]
    _xdev = Xexp[i*21 : (i+1)*21, :]
    Xorigin.append(np.vstack([_x, _xdev]))



rmse = []
for train_size in train_sizes:
    print('Train Size = %d' %train_size)
    test_size = 100 - train_size
    fp = open('./result/Dopt_new/result_%d.txt' %(train_size), 'w')
    fp.write('Input file: %s \n' %singlepts)
    fp.write('Train size: %d \n' %train_size)
    fp.write('Test size: %d \n' %test_size)
    fp.write('Starting for Design = %d\n' %nstart)
    fp.write('Initial training = %d\n' %ninit)
    fp.write('--'*20 + '\n')
    fp.write('Regressor: %s \n' %regr)
    fp.write('Scale in x, y: %f \n' %scale)
    fp.write('Degree of freedom in wave Cal : %d \n' %DOF)
    fp.write('Maximum order: %s \n' %order)
    fp.write('wave IB=7:' + str(wave_ib7.tolist()) + '\n')
    fp.write('==' * 20 + '\n')
    

    Design = D_opt(ntarget=train_size, Xorigin=Xorigin, n_job=2)
    index_select, crit = Design.multiSeq(nstart=nstart, ninit=ninit)

    # %% Retrieve the design data
    xtrain = [xfull[idx] for idx in index_select]
    ytrain = [yfull[idx] for idx in index_select]
    xdev_train = [xdev_full[idx] for idx in index_select]
    dydx_train = [dydx_full[idx] for idx in index_select]
    # add origin
    xorigin, yorigin, xdev_origin, dydx_origin = extract(origin, origin[0], shift=True, scale=scale)
    xtrain.append(xorigin[0])
    ytrain.append(yorigin[0])
    xdev_train.append(xdev_origin[0])
    dydx_train.append(dydx_origin[0])

    # %% FIT PES
    weights = 1
    pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, weights=weights)
    Hpce = pce.derivative(m=2) / (scale**2)
    fp.write('Index Select:' + str(index_select) + '\n')
    fp.write('Final Criterion:' + str(crit) + '\n')
    # Error in Hessian
    fp.write('Error in Hessian:' + str(error_cal(H_ib7, Hpce)) + '\n')
    # Calculate wave number
    freq_pce, wave_pce = vibcal(Hpce, atoms)
    fp.write('Error in Wave Number (ALL):' + str(error_cal(wave_ib7, wave_pce)) + '\n')
    fp.write('Error in Wave Number (DOF):' + str(error_cal(wave_ib7[:DOF], wave_pce[:DOF])) + '\n')
    fp.write('wave PCE :' + str(wave_pce.tolist()) + '\n')
    rmse.append(error_cal(wave_ib7[:DOF], wave_pce[:DOF])[0])
    fp.write('==' * 20 + '\n')
    fp.close()



import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_sizes, rmse, 'ro-')