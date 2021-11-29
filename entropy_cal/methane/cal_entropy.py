'''
TOTAL ENTROPY CALCULATION
'''

import numpy as np
import matplotlib.pyplot as plt

S2D = np.load('logint_methane_2D_63.npy')
Srot = np.load('logint_methane_rot_63.npy')
Svib = np.load('Svib_methane_63.npy').flatten()

plt.figure()
#plt.hist(Svib, bins=50, density=True, label='Sample', histtype='step')
#plt.hist(S2D, bins=50, density=True, label='Sample', histtype='step')
plt.hist(Srot, bins=50, density=True, label='Sample', histtype='step')

l = len(Svib)

Sconfig = 4.814211
Strans_base = 4.09540416
Srot_base = 0.341491


Sexp = 11.7

Stot = S2D[:l] + 1/2. * Srot[:l] + Svib + Sconfig + Strans_base + Srot_base

plt.figure()
m, _, _ = plt.hist(Stot, bins=50, density=True, label='Sample')
plt.plot([Sexp, Sexp], [0, 1.1*max(m)], 'k:', linewidth=3, label='Exp')
plt.plot([Stot.mean(), Stot.mean()], [0, 1.1*max(m)], 'r:', linewidth=3, label='Mean')


plt.legend(frameon=False)
plt.xlabel('INTEGRAL')
plt.ylabel('Probability density')
plt.tight_layout()


print('MEHANE @63K:')
print('Mean of total entropy = %.2f +\- %.2f' %(Stot.mean(), Stot.std()))

