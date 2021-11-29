# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:58:35 2019

@author: hut216
"""

# testing linear regression uncertainty quantification

import numpy as np

import statsmodels.api as sm

Y = [1,3,4,5,2,3,4]
X = range(1,8)
X = sm.add_constant(X)

model = sm.OLS(Y,X)
params = np.array([2.14285714, 0.25      ])

#results = model.fit()
