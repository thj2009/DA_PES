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
fld = './'
# Methane
natom = 5; nvar = 15; nord = 2
# Ethane
#natom = 8; nvar = 24; nord = 2

asym = Sampling(nvar=nvar, nord=nord, nsample=nsample, scale=0.005).asymptotic()
for _a in asym:
    print(np.linalg.norm(_a))


#print(np.max(asym))

GEO = []

for i in range(nsample):
    dev = asym[i, :]
    print(np.linalg.norm(dev))
    dev = dev.reshape((natom, 3))

#    fgen = open(os.path.join(fld, 'sample', 'ethane_0.015', 'POSCAR%d' %i), 'w')
    fgen = open(os.path.join(fld, 'sample', 'methane_0.015', 'POSCAR%d' %i), 'w')

    fp = open(os.path.join(fld, 'methane'), 'r')

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
        if idx == 7:
#            print(line)
            if line == 'Direct':
                # Fractional Geometry
                rescale = np.linalg.inv(box)
            elif line == 'Cartesian':
                # Cartesian Geometry
                rescale = np.eye(3)

        if idx >= 8:
            i_c = atomlist.index('C')
            num_other = np.sum(atom_number[0: i_c])
        if idx >= num_other + 8:
            # start reading geometry
            geo = [float(term) for term in line.split()[0:3]]
            _i = idx - (num_other + 8)
#            print(_i)
#            print(_i)
#            new_geo = np.array(geo) + rescale.dot(dev[_i, :])
            new_geo = np.array(geo).dot(np.linalg.inv(rescale))

#            print(_i, geo, new_geo)
#            line = '{0:<18.16f}  {1:<18.16f}  {2:<18.16f}   T   T   T'.format(new_geo[0], new_geo[1], new_geo[2])
            line = '{0:>13.5f}  {1:>13.5f}  {2:>13.5f} **'.format(new_geo[0], new_geo[1], new_geo[2])
            print(line)
#        print(np.linalg.norm(dev.flatten()))

        fgen.write(line + '\n')
    fp.close()
    fgen.close()
    pass







