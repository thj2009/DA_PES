# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 16:28:59 2019

@author: hut216
"""

import os
import numpy as np
from DA_PES import Sampling

nsample = 100
# Ronak Case
#fld = './ronak/'
#natom = 6; nvar = 18; nord = 2

## Chris
#fld = './chris'
#natom = 7; nvar = 21; nord = 2

# Srinivas
fld = './srinivas'
natom = 6; nvar = 18; nord = 2

asym = Sampling(nvar=nvar, nord=nord, nsample=nsample, scale=0.005).asymptotic()
for _a in asym:
    print(np.linalg.norm(_a))

'''
#print(np.max(asym))

for i in range(nsample):
    dev = asym[i, :]
    print(np.linalg.norm(dev))
    dev = dev.reshape((natom, 3))
#    fgen = open(os.path.join(fld, 'sample', 'test_0.005', 'POSCAR%d' %i), 'w')
    fgen = open(os.path.join(fld, 'sample', 'test_0.015', 'POSCAR%d' %i), 'w')

    fp = open(os.path.join(fld, 'POSCAR'), 'r')

    box = []
    atomlist = []
    atom_number = []
    num_other = 0
    for idx, line in enumerate(fp.readlines()):
        line = line.strip()
#        print(line)
        if 'Selective dynamics' in line:
            continue
        # extract knowledge for last atoms
        if idx == 0:
            pass
        if idx in [2, 3, 4]:
            box.append([float(term) for term in line.split()])
        if idx == 5:
            atomlist = [term for term in line.split()]
        if idx == 6:
            atom_number = [int(term) for term in line.split()]
        if idx == 8:
            print(line)
            if line == 'Direct':
                # Fractional Geometry
                rescale = np.linalg.inv(box)
            elif line == 'Cartesian':
                # Cartesian Geometry
                rescale = np.eye(3)

        if idx >= 9:
            i_c = atomlist.index('C')
            num_other = np.sum(atom_number[0: i_c])
        if idx >= num_other + 9:
            # start reading geometry
            geo = [float(term) for term in line.split()[0:3]]
            _i = idx - (num_other + 9)
#            print(_i)
            new_geo = np.array(geo) + rescale.dot(dev[_i, :])
#            print(_i, geo, new_geo)
#            line = '{0:<18.16f}  {1:<18.16f}  {2:<18.16f}   T   T   T'.format(new_geo[0], new_geo[1], new_geo[2])
            line = '{0:<18.16f}  {1:<18.16f}  {2:<18.16f}'.format(new_geo[0], new_geo[1], new_geo[2])

#        print(np.linalg.norm(dev.flatten()))

        fgen.write(line + '\n')
    fp.close()
    fgen.close()
    pass

'''






