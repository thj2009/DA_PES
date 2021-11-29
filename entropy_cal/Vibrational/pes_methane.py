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
fld = 'methane_015'

singlepts = os.path.join(fld, 'energy-forces.txt')


regr = 'OLS'
order = 2
scale = 1
polytype = 'Herm'

atoms = ['C', 'H', 'H', 'H', 'H']

np.random.seed(seed)
nd, datalist = read_data(singlepts, 5)
_, origin = read_data(os.path.join(fld, 'origin.txt'), 5)

train_size = 90
rmse, std = [], []
pce = DA_PES.PCE(15, order, withdev=True, copy_xy=True, regr=regr, polytype=polytype)

data_train, data_test = train_test_split(datalist, test_size=100-train_size)

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


DOF = 12
nsample = 50000
wave_list = []
Hlist = pce.uq_2nd_order(nsample=nsample)
for _H in Hlist:
    _freq, _wave = vibcal(_H / (scale**2) , atoms)
    wave_list.append(_wave)

wave_mean = np.mean(wave_list, axis=0)
wave_std = np.std(wave_list, axis=0)

Tem = 63

kb = 1.38064852e-23 # m2 kg s-2 K-1
h = 6.62607004e-34 # m2 kg / s
c = 299792458 # m / s



Srot_base = 0.341491
S_config = 4.814211
S2D_base = 4.09540416


dfpt =read_wave('./methane_015/dfpt.txt')

# =============================================================================
# THRESHOLD
# =============================================================================
thres = 50
#Tems = np.arange(40, 150, 0.5)
Tems = np.array([63])
Slist = []
Sind_list = []
for i in range(nsample):
    _wave = wave_list[i][:DOF]
    if 0 not in _wave:
#    if True:
#        _wave = [max(w, thres) for w in _wave]
    #    _wave[-1] = 50
        eps = np.array(_wave) * 100 * c * h
        beta = 1. / (kb * Tems)
        term = np.exp(-np.outer(beta, eps))
        SS = -np.log(1 - term) + np.outer(beta, eps) / (term**(-1) - 1)
    #    print(SS.T.flatten().shape)
        Sind_list.append(SS.T.flatten().tolist())
        Svib = np.sum(SS, axis=1)
        Slist.append(Svib)


Slist = np.array(Slist).flatten()
print('Threshold = %.2f' %thres)
print('Surrogate Sample = ', np.mean(Slist), '+/-', np.std(Slist))
#print('Surrogate Sample = ', np.median(Slist), '+/-', np.std(Slist))

np.save('Svib_methane_%d.npy' %Tem, Slist)

Sstar = np.array(Sind_list)
plt.figure()
for i in range(4):
    for j in range(3):
        plt.subplot(3, 4, 3*i+j+1)
        plt.hist(Sstar[:, 3*i+j])


# 2D-TRANSLATION
logQtrans = np.random.normal(-0.691, 0.016, (nsample, 1))
logQvib = 1/2. * np.random.normal(-0.032, 0.202, (nsample, 1))

Stot = Slist + logQtrans + logQvib

# PCE MEAN
_wave = [max(w, thres) for w in wave_pce[:DOF]]
eps = np.array(_wave) * 100 * c * h
beta = 1. / (kb * Tems)
term = np.exp(-np.outer(beta, eps))
SS = -np.log(1 - term) + np.outer(beta, eps) / (term**(-1) - 1)
Svib_pce = np.sum(SS, axis=1)
print('PCE wavenumber = ', wave_pce[:DOF])

# PCE MEAN
_wave = [max(w, thres) for w in dfpt[:DOF]]
eps = np.array(_wave) * 100 * c * h
beta = 1. / (kb * Tems)
term = np.exp(-np.outer(beta, eps))
SS = -np.log(1 - term) + np.outer(beta, eps) / (term**(-1) - 1)
Svib_dfpt = np.sum(SS, axis=1)


Stot_pce = Svib_pce + Srot_base + S_config + S2D_base + logQtrans + logQvib
Stot_dfpt = Svib_pce + Srot_base + S_config + S2D_base + logQtrans + logQvib


plt.figure()
plt.subplot(1,2,1)
#lb = 'mean = %.3f \n std = %.3f' %(np.mean(log_int), np.std(log_int))
m, _, _ = plt.hist(Slist, bins=50, density=True, label='Sample')
plt.plot([Svib_pce, Svib_pce], [0, 1.1*max(m)], 'k:', linewidth=3, label='PCE')
plt.plot([Svib_dfpt, Svib_dfpt], [0, 1.1*max(m)], 'r:', linewidth=3, label='Numerical')
plt.title('S_vib, Threshold = %.1e' %thres)
plt.legend(frameon=False)
plt.xlabel('INTEGRAL')
plt.ylabel('Probability density')
plt.tight_layout()


plt.subplot(1,2,2)
#lb = 'mean = %.3f \n std = %.3f' %(np.mean(log_int), np.std(log_int))
plt.hist(Stot + Srot_base + S_config + S2D_base, bins=50, density=True, label='Sample')
plt.hist(Stot_pce, bins=50, density=True, label='PCE')
plt.title('Stot, Threshold = %.1e' %thres)
plt.xlabel('INTEGRAL')
plt.ylabel('Probability density')
plt.tight_layout()

#full_wave = np.array(wave_list)
#plt.figure()
#for i in range(4):
#    for j in range(3):
#        plt.subplot(3, 4, 3*i+j+1)
#        m, _, _ = plt.hist(full_wave[:, 3*i+j], density=True)
#        plt.plot([dfpt[3*i+j], dfpt[3*i+j]], [0, max(m)*1.1], 'k:', linewidth=2)

# bar plot
plt.figure()
plt.bar(range(12), np.mean(Sind_list,axis = 0), yerr=np.std(Sind_list, axis=0))

