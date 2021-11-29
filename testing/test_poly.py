# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:47:50 2019

@author: huijie
"""

import DA_PES

h = DA_PES.Hermite(order=3)

print(h)
print(h.der(m=1))
print(h.der(m=2))
print(h.der(m=3))
print(h.der(m=4))
print(h.der(m=5))