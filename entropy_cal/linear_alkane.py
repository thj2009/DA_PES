'''
Charles T Campbell Entropy data
'''

import numpy as np
import matplotlib.pyplot as plt

Tems = np.array([63, 106, 139, 171, 229])
Sgas = np.array([16.1, 22.3, 26.5, 31.9, 42.8])
Sads = np.array([11.7, 14.2, 15.5, 21.0, 26.0])


S_m = 9.24; std_m = 0.40
S_e = 15.03; std_e = 0.73

_x = np.linspace(7,45, 100)
_y = -3.3 + 0.7 * _x
plt.figure(figsize=(3, 3), dpi=300)
plt.plot(Sgas, Sads, 'ro')
plt.plot(_x, _y, 'r:')

# linear fitting on Sgas & Sads
z = np.polyfit(Sgas, Sads, 1)
poly = np.poly1d(z)
_fit = poly(_x)
plt.plot(_x, _fit, 'k:')

plt.errorbar(Sgas[0], S_m, yerr=std_m, fmt='b^')
plt.errorbar(Sgas[1], S_e, yerr=std_e, fmt='b^')

plt.xlabel('Sgas/R', fontsize=8)
plt.ylabel('Sads/R', fontsize=8)

plt.xlim([7, 45])
plt.ylim([7,30])
plt.tight_layout()