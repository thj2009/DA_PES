# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:34:06 2019

pce test with fix training data
@author: huijie
"""

# Work on ethylene dataset

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal, sub_hist
from DA_PES.vibfreq import read_hess, vibcal, read_wave_dft
from DA_PES import D_opt
from sklearn.model_selection import train_test_split

#np.random.seed(10)
#nd, datalist = read_data('./output/test_0.045_chris.txt', 7)
nd, datalist = read_data('./output/test_0.015_chris.txt', 7)
#nd, datalist = read_data('./output/test_0.010_chris.txt', 7)
#nd, datalist = read_data('./output/test_symmetric.txt', 7)

_, origin = read_data('./output/origin.txt', 7)


#H_ib7 = read_hess('./output/hessian/IB7.txt', sym=True)
H_ib5nf1 = read_hess('./output/hessian/IB5NF1.txt', sym=True)
H_ib5nf2 = read_hess('./output/hessian/IB5NF2.txt', sym=True)
H_ib5nf4 = read_hess('./output/hessian/IB5NF4.txt', sym=True)


atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

#freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)
freq_ib5nf4, wave_ib5nf4 = vibcal(H_ib5nf4, atoms, real=False)
freq_ib5nf2, wave_ib5nf2 = vibcal(H_ib5nf2, atoms, real=False)

wave_nf4_dft = read_wave_dft('./output/hessian/frequencies_NFREE_4.txt')


print('{0:^5s}  {1:^9s}  {2:^9s}'.format('dof', 'dft', 'hess'))
print('=='*20)
for i, (w_dft, w_hess) in enumerate(zip(wave_nf4_dft, wave_ib5nf4)):
    if np.real(w_hess) != 0:
        w_hess = float(w_hess)
    print('{0:^5d}  {1:^9.4f}  {2:^9.4f}'.format(i+1, w_dft, w_hess))
    
err = np.array(wave_nf4_dft) - np.array(wave_ib5nf4)
pce = DA_PES.PCE(21, 3, withdev=True, copy_xy=True, regr='OLS',
                 polytype='Plain')

#train_size = 100
#test_size = 100 - train_size
#data_train, data_test = train_test_split(datalist, test_size=test_size)
##
#scale = 1
#xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
##xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)
#
#
## %% test EXP
##xfull, yfull, xdev_full, dydx_full = extract(datalist + origin, origin[0], shift=True, scale=scale)
#nd = len(xtrain)
##import matplotlib.pyplot as plt
##for i in range(21):
##    plt.figure()
##    plt.hist(xfull[:, i])
#weights = 1
##weights = [0] * nd +[1] * nd * 21
##pce.fit(xfull, yfull, xdev=xdev_full, dydx=dydx_full, uncer=True, njob=2, weights=weights)
#pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)
#
#Hpce = pce.derivative(m=2) / (scale**2)
#
#DOF = 15
## Calculate wave number
atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']
#
#wave_list = []
#Hlist = pce.uq_2nd_order(nsample=1000)
#for _H in Hlist:
#    _freq, _wave = vibcal(_H / (scale**2) , atoms)
#    wave_list.append(_wave)
#
#wave_mean = np.mean(wave_list, axis=0)
#wave_std = np.std(wave_list, axis=0)
#
##sub_hist(np.array(wave_list)[:, :DOF], [4,4])
#
#
## Error in Hessian
#print('Error in Hessian: \n' + '=='*20)
#print(error_cal(H_ib7, Hpce))
#print(error_cal(H_ib7, H_ib5nf1))
#print(error_cal(H_ib7, H_ib5nf2))
#print(error_cal(H_ib7, H_ib5nf4))
#
#
#
#
#freq_pce, wave_pce = vibcal(Hpce, atoms)
#freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)
#freq_ib5nf4, wave_ib5nf4 = vibcal(H_ib5nf4, atoms)
#freq_ib5nf2, wave_ib5nf2 = vibcal(H_ib5nf2, atoms)
#
#print('{0:^6s}  {1:^12s}  {2:^12s}  {3:^12s}'.format('#', 'wave(ib7)', 'mean', 'std'))
#print('=='*50)
#for i in range(21):
#    print('{0:^6d}  {1:^12.2f}  {2:^12.2f}  {3:^12.2f}'.format(i, wave_ib7[i], wave_mean[i], wave_std[i]))
#
#
#print('\n\n\n')
#print('{0:^15s}  {1:^12s}  {2:^12s}  {3:^12s}  {4:^12s}  {5:^12s}  {6:^12s}'.format('#', 'wave(ib7)', 'wave(nf2)', 'wave(nf4)', 'pce', 'mean', 'std'))
#print('=='*50)
#for i in range(21):
#    print('{0:^15d}  {1:^12.2f}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12.2f}'.format(i, wave_ib7[i], wave_ib5nf2[i], wave_ib5nf4[i], wave_pce[i], wave_mean[i], wave_std[i]))
#
#
##rmse_nf1, _ = error_cal(wave_ib7[:DOF], wave_ib5nf1[:DOF])
#rmse_nf2, _, rel_nf2 = error_cal(wave_ib7, wave_ib5nf2)
#rmse_nf4, _, rel_nf4 = error_cal(wave_ib7, wave_ib5nf4)
#rmse_pce, _, rel_pce = error_cal(wave_ib7, wave_pce)
#rmse_pce_sample, _, rel_pce_sample = error_cal(wave_ib7, wave_mean)
#
#print('--'*50)
#print('{0:^15s}  {1:^12s}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12s}'.format('RMSE(full)', '--', rmse_nf2, rmse_nf4, rmse_pce, rmse_pce_sample, '--'))
#print('{0:^15s}  {1:^12s}  {2:^12.2e}  {3:^12.2e}  {4:^12.2e}  {5:^12.2e}  {6:^12s}'.format('Relerr(full)', '--', rel_nf2, rel_nf4, rel_pce, rel_pce_sample, '--'))
#
##rmse_nf1, _, rel_nf1 = error_cal(wave_ib7[:DOF], wave_ib5nf1[:DOF])
#rmse_nf2, _, rel_nf2 = error_cal(wave_ib7[:DOF], wave_ib5nf2[:DOF])
#rmse_nf4, _, rel_nf4 = error_cal(wave_ib7[:DOF], wave_ib5nf4[:DOF])
#rmse_pce, _, rel_pce = error_cal(wave_ib7[:DOF], wave_pce[:DOF])
#rmse_pce_sample, _, rel_pce_sample = error_cal(wave_ib7[:DOF], wave_mean[:DOF])
#print('\nError (DOF) in WaveNumber')
#print('--'*50)
#
#print('{0:^15s}  {1:^12s}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12s}'.format('RMSE(dof)', '--', rmse_nf2, rmse_nf4, rmse_pce, rmse_pce_sample, '--'))
#print('{0:^15s}  {1:^12s}  {2:^12.2e}  {3:^12.2e}  {4:^12.2e}  {5:^12.2e}  {6:^12s}'.format('Relerr(dof)', '--', rel_nf2, rel_nf4, rel_pce, rel_pce_sample, '--'))
#
##print(Hpce)
#
#fp = open('HPCE.txt', 'w')
#for i in range(21):
#    for j in range(21):
#        fp.write('{0:^12.5f}'.format(Hpce[i, j]))
#    fp.write('\n')
#fp.close()