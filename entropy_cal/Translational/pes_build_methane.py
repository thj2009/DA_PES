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
from DA_PES.vibfreq import read_hess, vibcal, center_of_mass
from sklearn.model_selection import train_test_split

#np.random.seed(10)
nd, fulllist = read_data('./methane/energy-forces-sampling.txt', 5)

_, origin = read_data('./methane/energy-forces-min.txt', 5)

scale = 1

# Shift xfull
for data in fulllist:
    for i in range(5):
        data['coord'][i][2] += 8.6

xfull, yfull, xdev_full, dydx_full = extract(fulllist + origin, origin[0], shift=True, scale=scale)


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xfull[:, 0], xfull[:,1])

atoms = ['C', 'H', 'H', 'H', 'H']
# Calculate center of mass
xfull = center_of_mass(atoms, xfull)
#xfull = np.copy(xfull[:, :2])

dydx_t = []
for dydx in dydx_full:
    force = dydx.reshape([5, 3])
    force = np.sum(force, axis=0)
#    force = np.copy(force[:2])
    dydx_t.append(force)
    
pce = DA_PES.PCE(3, 5, withdev=True, copy_xy=True, regr='OLS', polytype='Plain')
#pce = DA_PES.PCE(2, 10, withdev=True, copy_xy=True, regr='OLS', polytype='Plain')

xtrain, xtest, ytrain, ytest, xdev_train, xdev_test, dydx_train, dydx_test = \
        train_test_split(xfull, yfull, xfull, dydx_t, test_size=20)

nt = len(xtrain)
weights = [1] *nt + [0.1] * 3 * nt
#weights = [1] *nt + [0.1] * 2 * nt

#weights = 1
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
print(is_pos_def(pce.cov_param))
## =============================================================================
## Predict on TEST set
## =============================================================================
#
#
ytest_pred = pce.predict(xtest)
parity(ytest, ytest_pred)
print(error_cal(ytest, ytest_pred))

dydx_test_pred = []
for _x in xtest:
    dydx_test_pred.append(pce.derivative(m=1, x=_x))
parity(dydx_test, dydx_test_pred)
print(error_cal(dydx_test, dydx_test_pred))





# =============================================================================
# INTEGRATION
# SAMPLING
# =============================================================================

DX = np.array([1.46, -0.872, 0])
DY = np.array([1.46, 0.872, 0])

Nsample = 100

xsample = []
for i in range(Nsample):
    for j in range(Nsample):
        dxy = DX * i / Nsample + DY * j /Nsample
        if dxy[0] <= 1.46:
            xsample.append(dxy)
xsample = np.array(xsample)
nsample = len(xsample)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xsample[:, 0], xsample[:,1])

n_sam = 50000
ysample = pce.predict(xsample)
ypred_sample = pce.sample(xsample, n_sam)

# check mean

#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#
#X = np.arange(Nsample)
#Y = np.arange(Nsample)
#X, Y = np.meshgrid(X, Y)
#Z = np.reshape(np.copy(ysample), [Nsample, Nsample])
#
## Plot the surface.
#surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)



from DA_PES.utils import Constant as _const
from DA_PES.utils import Converter as _conv

Tem = 63 # K
AREA = np.cross(DX[:2], DY[:2])/2. * _conv.A_2_m **2
INTE = np.sum(np.exp((-ysample * _conv.eV_2_J)/(_const.kB * Tem))) / (nsample) 
PREFC = 2 * np.pi * _const.kB * Tem / (_const.h ** 2) * (12.011 + 4) * _conv.u_2_kg
#SUMM = np.sum(np.exp((-np.array(E) * _conv.eV_2_J)/(_const.kB * Tem))) / (len(E)) * (AREA * _conv.A_2_m **2)

# =============================================================================
# Calculate the integral with sampling and UQ
# =============================================================================

#Tems = np.arange(40, 150, 0.5)
Tems = 63

Zlist = []
#for Tem in Tems:
inte_sample = []
for i in range(n_sam):
    inte = np.sum(np.exp((-ypred_sample[:,i] * _conv.eV_2_J)/(_const.kB * Tem))) / (nsample) 
    inte_sample.append(inte)
#mean_int = np.mean(inte_sample)
#std_int = np.std(inte_sample)
log_int = np.log(inte_sample)
#    Zlist.append()
#plt.figure()
#plt.hist(inte_sample, bins=50, density=True)
#plt.xlabel('INTEGRAL')
#plt.ylabel('Probability density')
#plt.tight_layout()
np.save('logint_methane_2D_%d.npy' %Tem, log_int)

#
plt.figure()
lb = 'mean = %.3f \n std = %.3f' %(np.mean(log_int), np.std(log_int))
plt.hist(log_int, bins=50, density=True, label=lb)
plt.legend(frameon=False)
plt.xlabel('INTEGRAL')
plt.ylabel('Probability density')
plt.tight_layout()
#
print('==' * 20)
print('Temperature = %d K' %Tem)
print('AREA = %e' %AREA)
print('INTEGRAL = %e' %INTE)
print('  UQ: log(INTEGRAL) = %.3e +/-  %.3e' %(np.mean(log_int), np.std(log_int)))
print('PREFACTOR = %e' %PREFC)