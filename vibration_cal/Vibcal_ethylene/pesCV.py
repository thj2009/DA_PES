# ten fold cross validation
import numpy as np
import DA_PES
from DA_PES.utils import read_data, extract
from DA_PES.utils import parity, error_cal, weight_construct
from DA_PES.vibfreq import read_hess, vibcal
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



regr = 'OLS'
order = 2
scale = 1
polytype = 'Herm'


# read data
scl = 0.015
singlepts = './output/test_%.3f_chris.txt' %scl
nd, datalist = read_data(singlepts, 7)
_, origin = read_data('./output/origin.txt', 7)


fold = 5
kf = KFold(n_splits=fold, shuffle=True)

YPRED, YFORCEPRED = [], []
Y, YFORCE = [], []
for train_index, test_index in kf.split(datalist):
    data_train = [datalist[i] for i in train_index]
    data_test = [datalist[i] for i in test_index]

    xtrain, ytrain, xdev_train, dydx_train = extract(data_train + origin, origin[0], shift=True, scale=scale)
    xtest, ytest, xdev_test, dydx_test = extract(data_test, origin[0], shift=True, scale=scale)

    pce = DA_PES.PCE(21, order, withdev=True, copy_xy=True, regr=regr, polytype=polytype)
    weights = 1
    pce.fit(xtrain, ytrain, xdev=xdev_train, dydx=dydx_train, uncer=True, njob=2, weights=weights)

    ytest_pred = pce.predict(xtest).tolist()
    # force
    ydevtest_pred = []
    for _x in xdev_test:
        ydevtest_pred.append(pce.derivative(m=1, x=_x).tolist())

    YPRED.append(ytest_pred)
    YFORCEPRED.append(ydevtest_pred)
    Y.append(ytest)
    YFORCE.append([dy.tolist() for dy in dydx_test])
data = {}
data['y_pred'] = YPRED
data['yforce_pred'] = YFORCEPRED
data['y'] = Y
data['yforce'] = YFORCE

import json
with open('./result/CV_%d.txt' %fold, 'w') as fp:
    json.dump(data, fp, indent=True)
