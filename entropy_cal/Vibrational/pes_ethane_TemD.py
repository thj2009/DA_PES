# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:15:40 2019

@author: huijie
"""

import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal, weight_construct
from DA_PES.vibfreq import read_wave, vibcal
from sklearn.model_selection import train_test_split
import os
import json
import matplotlib.pyplot as plt
# =============================================================================
# SETTING
# =============================================================================

seed = np.random.randint(1000)
fld = 'ethane_015'

singlepts = os.path.join(fld, 'energy-forces.txt')


regr = 'OLS'
order = 2
scale = 1
polytype = 'Herm'

atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

np.random.seed(seed)
nd, datalist = read_data(singlepts, 8)
_, origin = read_data(os.path.join(fld, 'origin.txt'), 8)

DOF = 21
train_size = 95
rmse, std = [], []
pce = DA_PES.PCE(24, order, withdev=True, copy_xy=True, regr=regr, polytype=polytype)

data_train, data_test = train_test_split(datalist, test_size=100-train_size,
                                         random_state=np.random.randint(3000))

xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)


weights = 1
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)

ytest_pred = pce.predict(xtest)

dydx_pred = [pce.derivative(m=1, x=_x) for _x in xtest]
parity(ytest, ytest_pred)

parity(dydx_test, dydx_pred)

#    parity(dydx_test, dydx_pred)
Hpce = pce.derivative(m=2) / (scale**2)


freq_pce, wave_pce = vibcal(Hpce, atoms)


DOF = 21
nsample = 5000
wave_list = []
Hlist = pce.uq_2nd_order(nsample=nsample)
for _H in Hlist:
    _freq, _wave = vibcal(_H / (scale**2) , atoms)
    wave_list.append(_wave)
#
#wave_mean = np.mean(wave_list, axis=0)
#wave_std = np.std(wave_list, axis=0)

Tem = 106

kb = 1.38064852e-23 # m2 kg s-2 K-1
h = 6.62607004e-34 # m2 kg / s
c = 299792458 # m / s

Tems = np.arange(60, 160, 5)
#Tems = np.array(106)

Stot = []

for Tem in Tems:

    Slist = []
    #Zfull = []
    for i in range(nsample):
        _wave = wave_list[i][:DOF]
        if 0 not in _wave:
            eps = np.array(_wave) * 100 * c * h
            beta = 1. / (kb * Tem)
            term = np.exp(-np.outer(beta, eps))
            SS = -np.log(1 - term) + np.outer(beta, eps) / (term**(-1) - 1)
            Svib = np.sum(SS.flatten())
            Slist.append(Svib)
    Stot.append(Slist)
#Slist = np.array(Slist)
#print(np.mean(Slist), '+/-', np.std(Slist))

#np.save('Svib_ethane_%d.npy' %Tem, Slist)
np.save('Svib_enthane_Tem.npy', Stot)

Stot = np.array(Stot)
plt.figure()
plt.plot(Tems, np.mean(Stot, axis=1))


#plt.figure()
#plt.subplot(1,2,1)
##lb = 'mean = %.3f \n std = %.3f' %(np.mean(log_int), np.std(log_int))
#m, _, _ = plt.hist(Slist, bins=50, density=True, label='Sample')
##plt.plot([Svib_pce, Svib_pce], [0, 1.1*max(m)], 'k:', linewidth=3, label='PCE')
##plt.plot([Svib_dfpt, Svib_dfpt], [0, 1.1*max(m)], 'r:', linewidth=3, label='Numerical')
##plt.title('S_vib, Threshold = %.1e' %thres)
#plt.legend(frameon=False)
#plt.xlabel('INTEGRAL')
#plt.ylabel('Probability density')
#plt.tight_layout()

#data_dir = {}
#data_dir['Tempearture'] = Tems.tolist()
#data_dir['DOF'] = DOF
#data_dir['Entropy_list'] = Slist.tolist()
#with open('Svib_ethane.txt', 'w') as fp:
#    json.dump(data_dir, fp, indent=4)
##
##
#plt.figure()
#plt.plot(Tems, np.array(Slist).T)
#

#Tem = 106
#_wave = wave_pce[:DOF]
#print(_wave)
##_wave[-1] = 10
#
#thres = 20
##if 0 not in _wave:
#_wave = [max(w, thres) for w in _wave]
#eps = np.array(_wave) * 100 * c *h
#beta = 1. / (kb * Tem)
#term = np.exp(-np.outer(beta, eps))
#SS = -np.log(1 - term) + np.outer(beta, eps) / (term**(-1) - 1)
#Svib = np.sum(SS, axis=1)
#print(Svib)

