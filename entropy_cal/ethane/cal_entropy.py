'''
TOTAL ENTROPY CALCULATION
'''

import numpy as np
import matplotlib.pyplot as plt

S2D = np.load('logint_ethane_2D_106.npy')
Srot = np.load('logint_ethane_rot_106.npy')
Svib = np.load('Svib_ethane_106.npy').flatten()

plt.figure()
#plt.hist(Svib, bins=50, density=True, label='Sample', histtype='step')
#plt.hist(S2D, bins=50, density=True, label='Sample', histtype='step')
plt.hist(Srot, bins=50, density=True, label='Sample', histtype='step')


l = len(Svib)

Sconfig = 5.162265559
Strans_base = 5.24431719
Srot_base = 2.673009895


Sexp = 14.2

Stot = S2D[:l] + Srot[:l] + Svib + Sconfig + Strans_base + Srot_base

plt.figure()
m, _, _ = plt.hist(Stot, bins=50, density=True, label='Sample')
plt.plot([Sexp, Sexp], [0, 1.1*max(m)], 'k:', linewidth=3, label='Exp')
plt.plot([Stot.mean(), Stot.mean()], [0, 1.1*max(m)], 'r:', linewidth=3, label='Mean')


plt.legend(frameon=False)
plt.xlabel('INTEGRAL')
plt.ylabel('Probability density')
plt.tight_layout()

print('EHANE @106K:')
print('Mean of total entropy = %.2f +\- %.2f' %(Stot.mean(), Stot.std()))

