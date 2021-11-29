# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:14:30 2019

@author: huijie
"""
import matplotlib.pyplot as plt
import numpy as np

import os
# Analyst Result

DOF = 15

#flds = ['0.015_Plain', '0.015_Herm']
#flds = ['0.045', '0.015', '0.010']

scl = '0.015'
#train_sizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
#train_sizes = [30, 42, 52, 63, 73, 84, 94, 100]
flds = ['0.015', 'Dopt']
train_sizes = [30, 42, 52, 63, 73, 84]
fmts = ['kD-', 'bo-', 'g*-', 'r^-']

rmse_all, std_all = [], []
rel_all, rel_std_all = [], []
rmseDOF_all, rmseDOF_std_all = [], []
relDOF_all, relDOF_std_all = [], []

wave7list = []
for fld in flds:
    _rmse, _std = [], []
    _rel, _relstd = [], []
    _rmseDOF, _stdDOF = [], []
    _relDOF, _relDOFstd = [], []
    for train_size in train_sizes:
        file = os.path.join('result', fld, 'result_%d.txt' %train_size)
        fp = open(file, 'r')
        
        rmselist, rellist= [], []
        rmseDOFlist, relDOFlist = [], []
        for line in fp.readlines():
            line = line.strip()
            if 'Mean' in line:
                mean = float(line.split(':')[1])
            if 'Std' in line:
                std = float(line.split(':')[1])
            if 'wave IB=7' in line:
                splitline = line.split(':')[1][1:-1]
                wave7 = [float(t) for t in splitline.split(',')]
            if 'wave PCE' in line:
                splitline = line.split(':')[1][1:-1]
                wave_pce = [float(t) for t in splitline.split(',')]
                
                # calculate rmse
                err = np.array(wave7) - np.array(wave_pce)
                rmse = np.sqrt(np.mean(err ** 2))
                relerr = np.linalg.norm(err) / np.linalg.norm(np.array(wave7))
                rmsedof = np.sqrt(np.mean(err[:DOF] ** 2))
                relerrdof = np.linalg.norm(err[:DOF]) / np.linalg.norm(np.array(wave7)[:DOF])
                rmselist.append(rmse)
                rellist.append(relerr)
                rmseDOFlist.append(rmsedof)
                relDOFlist.append(relerrdof)
                
        print('scale = %5s' %scl)
        print('Number of training = %d' %train_size)
        print('RMSE: ', rmselist)
        print('--'*60)

        # RMSE on all DOF
        _rmse.append(np.mean(rmselist))
        _std.append(np.std(rmselist))
        # Rel err on all DOF
        _rel.append(np.mean(rellist))
        _relstd.append(np.std(rellist))
        # RMSE on select DOF
        _rmseDOF.append(np.mean(rmseDOFlist))
        _stdDOF.append(np.std(rmseDOFlist))
        # Rel err on select DOF
        _relDOF.append(np.mean(relDOFlist))
        _relDOFstd.append(np.std(relDOFlist))
        # wave 7
        wave7list.append(wave7)
        fp.close()
    rmse_all.append(_rmse); std_all.append(_std)
    rel_all.append(_rel); rel_std_all.append(_relstd)
    rmseDOF_all.append(_rmseDOF); rmseDOF_std_all.append(_stdDOF)
    relDOF_all.append(_relDOF); relDOF_std_all.append(_relDOFstd)


fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
ax = fig.gca()
for i in range(len(flds)):
    ax.errorbar(train_sizes, rmse_all[i], yerr=std_all[i], label=flds[i],
                fmt=fmts[i], markersize=4, elinewidth=1.5)
# ticks parameter size
ax.tick_params(axis='both', which='minor', direction='in', labelsize=7)
ax.tick_params(axis='both', which='major', direction='in', labelsize=7)

ax.set_xlabel('Training', fontsize=8)
ax.set_ylabel('RMSE in wave (cm-1)', fontsize=8)
#ax.set_yscale('log')
ax.legend(fontsize=6)
fig.tight_layout()


fig = plt.figure(figsize=(2.5, 2.5), dpi=300)
ax = fig.gca()
for i in range(len(flds)):
    ax.errorbar(train_sizes, rmseDOF_all[i], yerr=rmseDOF_std_all[i], label=flds[i],
                fmt=fmts[i], markersize=4, elinewidth=1.5)
# ticks parameter size
ax.tick_params(axis='both', which='minor', direction='in', labelsize=7)
ax.tick_params(axis='both', which='major', direction='in', labelsize=7)

ax.set_xlabel('Training', fontsize=8)
ax.set_ylabel('RMSE in wave (cm-1)', fontsize=8)
#ax.set_yscale('log')
ax.legend(fontsize=6)
fig.tight_layout()


for i in range(len(flds)):
    print('\n\nScales = %s' %flds[i])
    print('==' * 60)
    print('{0:^15s}  {1:^15s}  {2:^15s}  {3:^15s}  {4:^15s}  {5:^15s}  {6:^15s}'.\
          format('# of train', 'rmse', 'std', 'rmse(DOF)', 'std(DOF)', 'rel', 'rel(DOF)'))
    for j, train_size in enumerate(train_sizes):
            print('{0:^15d}  {1:^15.2f}  {2:^15.2f}  {3:^15.2f}  {4:^15.2f}  {5:^15.2e}  {6:^15.2e}'.\
                  format(train_size, rmse_all[i][j], std_all[i][j], rmseDOF_all[i][j], rmseDOF_std_all[i][j], rel_all[i][j], relDOF_all[i][j]))
