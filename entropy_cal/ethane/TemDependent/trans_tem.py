import numpy as np
import matplotlib.pyplot as plt


alpha = np.load('ethane_trans_Tem.npy')
Tems = np.arange(40, 320, 10)


fig = plt.figure(figsize=(2,2), dpi=300)
ax = fig.gca()

ax.plot(Tems, np.percentile(alpha, 50, axis=1), 'b', linewidth=0.8)
ax.fill_between(Tems,
                np.percentile(alpha, 84, axis=1),
                np.percentile(alpha, 16, axis=1),
                color='b', alpha=0.3)

ax.plot(Tems, np.zeros_like(Tems), 'r--')
ax.text(66, 0.02, 'Free Translator', color='r', fontsize=6)

ax.tick_params(axis='both', which='minor', direction='in', labelsize=6)
ax.tick_params(axis='both', which='major', direction='in', labelsize=6)


ax.set_xlim([50, 300])
ax.set_ylim([-0.7, 0.08])

ax.set_yticks([-0.6, -0.4, -0.2, 0.0])

ax.set_xlabel('Temperature (K)', fontsize=7)
ax.set_ylabel(r'$(\ln\alpha+\frac{d\ln\alpha}{d\lnT})/R$', fontsize=7)

fig.subplots_adjust(top=0.944,
bottom=0.216,
left=0.24,
right=0.94,
hspace=0.2,
wspace=0.2)

fig.savefig('alpha_trans_tem.pdf')