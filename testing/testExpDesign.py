# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:20:53 2019

@author: huijie
"""

# test experimental design
from DA_PES import D_opt
import numpy as np

#np.random.seed(10)
m = 10;
ntotal = 3000

Xorigin = np.random.uniform(-1, 1, [ntotal, m]).tolist()


Design = D_opt(ntarget=20, Xorigin=Xorigin)
index_select, crit = Design.sequential(ninit=1)
print(index_select)
print(crit)


index_select, crit = Design.multiSeq()
print(index_select)
print(crit)


# Compare with random selected datapoints
ninit = 20
index_select = index[:ninit].tolist()

index = np.arange(ntotal).tolist()

X = np.array([Xorigin[idx] for idx in index_select])
info = X.T.dot(X)
print(np.linalg.matrix_rank(info))
print(np.linalg.slogdet(info))



