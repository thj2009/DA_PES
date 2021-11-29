
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:17:45 2019

@author: thj2009
"""

# testing second order polynomial expansion model
import pytest
import DA_PES
import numpy as np
from DA_PES.utils import parity
p = 4


x = np.random.uniform(-1,1, (8, 2))
y = [np.exp(_x[0]) * np.sin(_x[1]) for _x in x]
dydx = [[np.exp(_x[0])*np.sin(_x[1]),
         np.exp(_x[0])*np.cos(_x[1])] for _x in x]

xtest = np.random.uniform(-1,1, (100, 2))
ytest = [np.exp(_x[0]) * np.sin(_x[1]) for _x in xtest]

pce = DA_PES.PCE(2, p, withdev=False, copy_xy=True)
#
pce.fit(x, y)
ypred = pce.predict(xtest)
#
#
parity(ytest, ypred)


pce_der = DA_PES.PCE(2, p, withdev=True, copy_xy=True)


pce_der.fit(x, y, xdev=x, dydx=dydx)
ypred = pce_der.predict(xtest)

parity(ytest, ypred)