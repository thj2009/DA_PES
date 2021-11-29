'''
TOTAL ENTROPY CALCULATION
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



Sconfig = 0
S50 = np.load('Svib_ethane_thres_50.0.npy') + Sconfig
S100 = np.load('Svib_ethane_thres_100.0.npy') + Sconfig
S150 = np.load('Svib_ethane_thres_150.0.npy') + Sconfig


Stot = np.load('Stot.npy')

fig = plt.figure(figsize=(3,3), dpi=300)
ax = fig.add_subplot(2,1,1)



ax.hist(S50, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='orange')
ax.hist(S100, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='g')
ax.hist(S150, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='r')
#ax.hist(Stot, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='b')


ax.hist(S50, density=True, bins=150, histtype='step', color='orange', label=r'Thres=50 $cm^{-1}$')
ax.hist(S100, density=True, bins=150, histtype='step', color='g', label=r'Thres=100 $cm^{-1}$')
ax.hist(S150, density=True, bins=150, histtype='step', color='r', label=r'Thres=150 $cm^{-1}$')
#ax.hist(Stot, density=True, bins=150, histtype='step', color='b')

ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.set_yticks([0, 2, 4, 6, 8])

ax.set_xlabel(r'$S^{frastrated}_{vib}/R$', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
ax.legend(frameon=False, fontsize=5.5)



ax.text(-0.13, 1.02, '(A)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)

ax.set_xlim([2, 9])


# =============================================================================
# NEGLECT less than threshold
# =============================================================================
Sconfig = 0
S50 = np.load('Svib_ethane_negthres_50.0.npy') + Sconfig
S100 = np.load('Svib_ethane_negthres_100.0.npy') + Sconfig
S150 = np.load('Svib_ethane_negthres_150.0.npy') + Sconfig


Stot = np.load('Stot.npy')

ax = fig.add_subplot(2,1,2)




ax.hist(S50, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='orange')
ax.hist(S100, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='g')
ax.hist(S150, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='r')
#ax.hist(Stot, density=True, bins=150, histtype='stepfilled', alpha=0.2, color='b')


ax.hist(S50, density=True, bins=150, histtype='step', color='orange', label=r'Thres=50 $cm^{-1}$')
ax.hist(S100, density=True, bins=150, histtype='step', color='g', label=r'Thres=100 $cm^{-1}$')
ax.hist(S150, density=True, bins=150, histtype='step', color='r', label=r'Thres=150 $cm^{-1}$')


ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.set_yticks([0, 2, 4, 6])

ax.set_xlabel(r'$S_{vib}/R$', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
ax.legend(frameon=False, fontsize=5.5)

ax.text(-0.13, 1.02, '(B)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)


ax.set_xlim([-0.2, 3.4])
ax.set_xticks([0, 1, 2, 3])

fig.subplots_adjust(top=0.951,
bottom=0.116,
left=0.116,
right=0.965,
hspace=0.379,
wspace=0.2)


fig.savefig('Dist_S_thres.pdf')