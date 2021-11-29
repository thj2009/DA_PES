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
from DA_PES.vibfreq import read_hess, vibcal
from DA_PES import D_opt
from sklearn.model_selection import train_test_split

#np.random.seed(10)
#nd, datalist = read_data('./output/test_0.045_chris.txt', 7)
nd, datalist = read_data('./output/test_0.015_chris.txt', 7)
#nd, datalist = read_data('./output/test_0.010_chris.txt', 7)

_, origin = read_data('./output/origin.txt', 7)


H_ib7 = read_hess('./output/hessian/IB7.txt', sym=True)
#H_ib5nf1 = read_hess('./output/hessian/IB5NF1.txt', sym=True)
H_ib5nf2 = read_hess('./output/hessian/IB5NF2.txt', sym=True)
H_ib5nf4 = read_hess('./output/hessian/IB5NF4.txt', sym=True)


pce = DA_PES.PCE(21, 2, withdev=True, copy_xy=True, regr='OLS')

#train_size = 84
#test_size = 100 - train_size
#data_train, data_test = train_test_split(datalist, test_size=test_size)
#
scale = 1
#xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
#xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)


# %% test EXP
xfull, yfull, xdev_full, dydx_full = extract(datalist, origin[0], shift=True, scale=scale)

#import matplotlib.pyplot as plt
#for i in range(21):
#    plt.figure()
#    plt.hist(xfull[:, i])
X = pce.expand_x(pce.order_list, xfull, 'Herm')
Xexp = pce.expand_dev_xy(pce.order_list, xdev_full, 'Herm')

# %% Construct Xorigin
Xorigin = []
for i in range(99):
    _x = X[i, :]
    _xdev = Xexp[i*21 : (i+1)*21, :]
    Xorigin.append(np.vstack([_x, _xdev]))


# %% Experimental design
Design = D_opt(ntarget=42, Xorigin=Xorigin, n_job=-1)
index_select, crit = Design.multiSeq(nstart=10)


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


nd = len(xtrain)
# %%

#weights = [1e-8] * nd + [1e-3] * 21 * nd
weights = 1
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)


Hpce = pce.derivative(m=2) / (scale**2)

DOF = 16
# Calculate wave number
atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H']

wave_list = []
Hlist = pce.uq_2nd_order(nsample=1000)
for _H in Hlist:
    _freq, _wave = vibcal(_H / (scale**2) , atoms)
    wave_list.append(_wave)

wave_mean = np.mean(wave_list, axis=0)
wave_std = np.std(wave_list, axis=0)

#sub_hist(np.array(wave_list)[:, :DOF], [4,4])


# Error in Hessian
print('Error in Hessian: \n' + '=='*20)
print(error_cal(H_ib7, Hpce))
#print(error_cal(H_ib7, H_ib5nf1))
print(error_cal(H_ib7, H_ib5nf2))
print(error_cal(H_ib7, H_ib5nf4))




freq_pce, wave_pce = vibcal(Hpce, atoms)
freq_ib7, wave_ib7 = vibcal(H_ib7, atoms)
freq_ib5nf4, wave_ib5nf4 = vibcal(H_ib5nf4, atoms)
freq_ib5nf2, wave_ib5nf2 = vibcal(H_ib5nf2, atoms)

print('{0:^6s}  {1:^12s}  {2:^12s}  {3:^12s}'.format('#', 'wave(ib7)', 'mean', 'std'))
print('=='*50)
for i in range(21):
    print('{0:^6d}  {1:^12.2f}  {2:^12.2f}  {3:^12.2f}'.format(i, wave_ib7[i], wave_mean[i], wave_std[i]))


print('\n\n\n')
print('{0:^15s}  {1:^12s}  {2:^12s}  {3:^12s}  {4:^12s}  {5:^12s}  {6:^12s}'.format('#', 'wave(ib7)', 'wave(nf2)', 'wave(nf4)', 'pce', 'mean', 'std'))
print('=='*50)
for i in range(21):
    print('{0:^15d}  {1:^12.2f}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12.2f}'.format(i, wave_ib7[i], wave_ib5nf2[i], wave_ib5nf4[i], wave_pce[i], wave_mean[i], wave_std[i]))


#rmse_nf1, _ = error_cal(wave_ib7[:DOF], wave_ib5nf1[:DOF])
rmse_nf2, _, rel_nf2 = error_cal(wave_ib7, wave_ib5nf2)
rmse_nf4, _, rel_nf4 = error_cal(wave_ib7, wave_ib5nf4)
rmse_pce, _, rel_pce = error_cal(wave_ib7, wave_pce)
rmse_pce_sample, _, rel_pce_sample = error_cal(wave_ib7, wave_mean)

print('--'*50)
print('{0:^15s}  {1:^12s}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12s}'.format('RMSE(full)', '--', rmse_nf2, rmse_nf4, rmse_pce, rmse_pce_sample, '--'))
print('{0:^15s}  {1:^12s}  {2:^12.2e}  {3:^12.2e}  {4:^12.2e}  {5:^12.2e}  {6:^12s}'.format('Relerr(full)', '--', rel_nf2, rel_nf4, rel_pce, rel_pce_sample, '--'))

#rmse_nf1, _, rel_nf1 = error_cal(wave_ib7[:DOF], wave_ib5nf1[:DOF])
rmse_nf2, _, rel_nf2 = error_cal(wave_ib7[:DOF], wave_ib5nf2[:DOF])
rmse_nf4, _, rel_nf4 = error_cal(wave_ib7[:DOF], wave_ib5nf4[:DOF])
rmse_pce, _, rel_pce = error_cal(wave_ib7[:DOF], wave_pce[:DOF])
rmse_pce_sample, _, rel_pce_sample = error_cal(wave_ib7[:DOF], wave_mean[:DOF])
print('\nError (DOF) in WaveNumber')
print('--'*50)

print('{0:^15s}  {1:^12s}  {2:^12.2f}  {3:^12.2f}  {4:^12.2f}  {5:^12.2f}  {6:^12s}'.format('RMSE(dof)', '--', rmse_nf2, rmse_nf4, rmse_pce, rmse_pce_sample, '--'))
print('{0:^15s}  {1:^12s}  {2:^12.2e}  {3:^12.2e}  {4:^12.2e}  {5:^12.2e}  {6:^12s}'.format('Relerr(dof)', '--', rel_nf2, rel_nf4, rel_pce, rel_pce_sample, '--'))
