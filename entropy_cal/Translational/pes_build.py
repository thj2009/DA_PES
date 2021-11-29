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

#np.random.seed(10)
nd, fulllist = read_data('./output/energy-forces-sampling.txt', 5)

_, origin = read_data('./output/energy-forces-min.txt', 5)

train_size = 100
scale = 1

pce = DA_PES.PCE(15, 2, withdev=True, copy_xy=True, regr='OLS', polytype='Plain')

datalist = []
for data in fulllist:
    dE = data['E0'] - origin[0]['E0']
    print(dE)
    if abs(dE) <= 0.01:
        datalist.append(data)
test_size = len(datalist) - train_size
data_train, data_test = train_test_split(datalist, test_size=test_size)


xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)

nt = len(xtrain)
weights = [10] * nt + [1] * nt * 15
pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)

# =============================================================================
# Predict on TEST set
# =============================================================================


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

#DX = np.array([1.37206, 0.     ])
#DY = np.array([0.68603, 1.18825])

DX = np.array([8.3155757467537992,  0.0000000000000000, 0])/6.
DY = np.array([4.1577878733768996,  7.2014998437825426, 0])/6.

Nsample = 100



xsample = []
for i in range(Nsample):
    for j in range(Nsample):
        dxy = DX * i / Nsample + DY * j /Nsample
        xsample.append(np.tile(dxy, 5))
xsample = np.array(xsample)


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(xsample[:, 0], xsample[:,1])


ysample = pce.predict(xsample)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = plt.figure()
ax = fig.gca(projection='3d')

X = np.arange(Nsample)
Y = np.arange(Nsample)
X, Y = np.meshgrid(X, Y)
Z = np.reshape(np.copy(ysample), [Nsample, Nsample])

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)



from DA_PES.utils import Constant as _const
from DA_PES.utils import Converter as _conv

Tem = 60 # K
AREA = np.cross(DX[:2], DY[:2])
# SUMM = np.sum(np.exp((-ysample * _conv.eV_2_J)/(_const.kB * Tem))) / (Nsample ** 2) /(AREA * _conv.A_2_m **2)
SUMM = np.sum(np.exp((-ysample * _conv.eV_2_J)/(_const.kB * Tem))) / (Nsample ** 2) * (AREA * _conv.A_2_m **2)

print(SUMM)

PREFC = 2 * np.pi * _const.kB * Tem / (_const.h ** 2) * (12.011 + 4) * _conv.u_2_kg
#Z = PREFC * SUMM
print(SUMM)
print(PREFC * SUMM)
# Direct Calculate the SUMM
#E = []
#for data in datalist:
#    dE = data['E0'] - origin[0]['E0']
#    E.append(dE)
#
#SUMM = np.sum(np.exp((-np.array(E) * _conv.eV_2_J)/(_const.kB * Tem))) / (len(E)) * (AREA * _conv.A_2_m **2)
#print(SUMM)
#print(PREFC * SUMM)