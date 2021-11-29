# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:15:40 2019

@author: huijie
"""

# test different scale
# on Chris dataset

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
repeat = 50
polytype = 'Legd'

atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

np.random.seed(seed)
nd, datalist = read_data(singlepts, 7)
_, origin = read_data('./output/origin.txt', 7)
H_ib7 = read_hess('./output/hessian/IB7.txt', sym=True)
freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)

#train_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
train_sizes = [30, 42, 52, 63, 73, 84, 94, 100]
#train_sizes = [27, 36]

rmse, std = [], []
for train_zize in train_sizes:
    print('Train Size = %d' %train_zize)
    test_size = 100 - train_zize
    fp = open('./result/%.3f/result_%d.txt' %(scl, train_zize), 'w')
    fp.write('Random Seed : %d \n' %seed)
    fp.write('Input file: %s \n' %singlepts)
    fp.write('Train size: %d \n' %train_zize)
    fp.write('Test size: %d \n' %test_size)
    fp.write('--'*20 + '\n')
    fp.write('Regressor: %s \n' %regr)
    fp.write('Scale in x, y: %f \n' %scale)
    fp.write('Degree of freedom in wave Cal : %d \n' %DOF)
    fp.write('Maximum order: %s \n' %order)
    fp.write('wave IB=7:' + str(wave_ib7.tolist()) + '\n')
    fp.write('Number of repeat: %d \n' %repeat)
    fp.write('TYPE of polynomial: %s \n' %polytype)
    fp.write('==' * 20 + '\n')
    
    pce = DA_PES.PCE(21, order, withdev=True, copy_xy=True, regr=regr, polytype=polytype)
    
    wave_rmse = []
    for i in range(repeat):
        print(i)
        data_train, data_test = train_test_split(datalist, test_size=test_size)
    
        xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
        xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)
        
#        weights = weight_construct(xtrain, 18)
        weights = 1
        pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)
        Hpce = pce.derivative(m=2) / (scale**2)
        fp.write('Iteration %d \n' %i)
        # Error in Hessian
        fp.write('Error in Hessian:' + str(error_cal(H_ib7, Hpce)) + '\n')
        # Calculate wave number
        freq_pce, wave_pce = vibcal(Hpce, atoms)
        fp.write('Error in Wave Number:' + str(error_cal(wave_ib7[:DOF], wave_pce[:DOF])) + '\n')
        fp.write('wave PCE :' + str(wave_pce.tolist()) + '\n')
        fp.write('---'*20 + '\n')
        wave_rmse.append(error_cal(wave_ib7[:DOF], wave_pce[:DOF])[0])
    
    fp.write('==' * 20 + '\n')
    fp.write('Wave Number RMSE: ' + str(wave_rmse) + '\n')
    fp.write('Mean : %.4f \n' %(np.mean(wave_rmse)))
    fp.write('Std : %.4f \n' %(np.std(wave_rmse)))
    fp.close()
    rmse.append(np.mean(wave_rmse))
    std.append(np.std(wave_rmse))
    
#    H_ib7 = read_hess('./output/hessian/IB7.txt', sym=True)
#    H_ib5nf1 = read_hess('./output/hessian/IB5NF1.txt', sym=True)
#    H_ib5nf2 = read_hess('./output/hessian/IB5NF2.txt', sym=True)
#    H_ib5nf4 = read_hess('./output/hessian/IB5NF4.txt', sym=True)
#    
#    freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)
#    freq_ib5nf4, wave_ib5nf4 = vibcal(H_ib5nf4, atoms)
#    freq_ib5nf2, wave_ib5nf2 = vibcal(H_ib5nf2, atoms)
#    freq_ib5nf1, wave_ib5nf1 = vibcal(H_ib5nf1, atoms)
    
#    # Error in Wave Number
#    print('Error in wave number: \n' + '=='*20)
#    print(error_cal(wave_ib7[:-DOF], wave_pce[:-DOF]))
#    print(error_cal(wave_ib7[:-DOF], wave_ib5nf1[:-DOF]))
#    print(error_cal(wave_ib7[:-DOF], wave_ib5nf2[:-DOF]))
#    print(error_cal(wave_ib7[:-DOF], wave_ib5nf4[:-DOF]))


import matplotlib.pyplot as plt
plt.figure()
plt.errorbar(train_sizes, rmse, yerr=std)