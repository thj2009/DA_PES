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
nd, fulllist = read_data('./ethane/energy-forces-sampling.txt', 8)

_, origin = read_data('./ethane/energy-forces-min.txt', 8)

scale = 1

# Shift xfull
for data in fulllist:
    for i in range(8):
        data['coord'][i][2] += 8.5

xfull, yfull, xdev_full, dydx_full = extract(fulllist + origin, origin[0], shift=True, scale=scale)


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xfull[:, 0], xfull[:,1])

atoms = ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
# Calculate center of mass
xfull = center_of_mass(atoms, xfull)

dydx_t = []
for dydx in dydx_full:
    force = dydx.reshape([8, 3])
    force = np.sum(force, axis=0)
    dydx_t.append(force)
    
pce = DA_PES.PCE(3, 5, withdev=True, copy_xy=True, regr='OLS', polytype='Herm')

xtrain, xtest, ytrain, ytest, xdev_train, xdev_test, dydx_train, dydx_test = \
        train_test_split(xfull, yfull, xfull, dydx_t, test_size=5)

nt = len(xtrain)
weights = [1] *nt + [0.1] * 3 * nt
#weights = 1
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)

## =============================================================================
## Predict on TEST set
## =============================================================================
#
#
ytest_pred = pce.predict(xtest)
parity(ytest, ytest_pred)
print(error_cal(ytest, ytest_pred))

#dydx_test_pred = []
#for _x in xtest:
#    dydx_test_pred.append(pce.derivative(m=1, x=_x))
#parity(dydx_test, dydx_test_pred)
#print(error_cal(dydx_test, dydx_test_pred))





# =============================================================================
# INTEGRATION
# SAMPLING
# =============================================================================

DX = np.array([1.46, -0.872, 0]) / scale
DY = np.array([1.46, 0.872, 0]) / scale

Nsample = 100

xsample = []
for i in range(Nsample):
    for j in range(Nsample):
        dxy = DX * i / Nsample + DY * j /Nsample
        if dxy[0] <= 1.46 / scale:
            xsample.append(dxy)
xsample = np.array(xsample)
nsample = len(xsample)

#import matplotlib.pyplot as plt
#plt.figure()
#plt.scatter(xsample[:, 0], xsample[:,1])

n_sam = 5000
ysample = pce.predict(xsample)
ypred_sample = pce.sample(xsample, nsample=n_sam)
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






Tems = np.arange(40, 320, 10)
Tot = []
for Tem in Tems:

    inte_sample = []
    inteTotSample = []
    for i in range(n_sam):
        inte = np.sum(np.exp((-ypred_sample[:,i] * _conv.eV_2_J)/(_const.kB * Tem))) / (nsample) 
        dinte_dT = np.sum(
                np.exp((-ypred_sample[:,i] * _conv.eV_2_J)/(_const.kB * Tem)) * 
                (ypred_sample[:,i]  * _conv.eV_2_J) /(_const.kB * Tem**2)
                ) / (nsample) 
        inte_sample.append(inte)
        inteTotSample.append(np.log(inte) + dinte_dT * Tem / inte)
    Tot.append(inteTotSample)

Tot = np.array(Tot)
plt.figure()
plt.fill_between(Tems, np.percentile(Tot, 84, axis=1), np.percentile(Tot, 16, axis=1), alpha=0.3, color='r')
plt.plot(Tems, np.mean(Tot, axis=1), color='r')


np.save('ethane_trans_Tem.npy', Tot)
# check positive definite

#def is_posdef(x):
#    posdef = np.all(np.linalg.eigvals(x) > 0)
#    symm = np.all(x == x.T)
#    print(posdef)
#    print(symm)
#    return posdef and symm
#
#print(is_posdef(pce.cov_param))