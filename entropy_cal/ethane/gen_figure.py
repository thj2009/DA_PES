'''
TOTAL ENTROPY CALCULATION
'''

import numpy as np
import matplotlib.pyplot as plt

S2D = np.load('logint_ethane_2D_106.npy')
Srot = np.load('logint_ethane_rot_106.npy')
Svib = np.load('Svib_ethane_106.npy').flatten()

Sconfig = 5.162265559
Strans_base = 5.24431719
Srot_base = 2.673009895

text = ''
text += '$S_{config}/R=%.1f$\n' %Sconfig
# =============================================================================
# Translational Entropy
# =============================================================================

fig = plt.figure(figsize=(3.25, 3.25), dpi=300)
ax = fig.add_subplot(2,2,1)

#fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
#ax = fig.gca()

n, _, _ = ax.hist(S2D+Strans_base, bins=60, density=True, linewidth=1, alpha=0.6, histtype='stepfilled')
ax.plot([Strans_base, Strans_base], [0, 9.5], 'r--', linewidth=1.4)
ax.text(5.02, 8.7, 'Free Translator', fontsize=5.5, color='r')
ax.set_xlabel(r'S$_{trans}$/R', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
#ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])

#ax.set_xlim([3.8, 6])
ax.set_ylim([0, 9.5])
ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.yaxis.set_label_coords(-0.2, 0.5)

ax.text(-0.18, 0.98, '(A)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)

#plt.subplots_adjust(top=0.95,
#                    bottom=0.18,
#                    left=0.18,
#                    right=0.95,
#                    hspace=0.2,
#                    wspace=0.2)
mean_ = np.mean(S2D+Strans_base)
std_ = np.std(S2D+Strans_base)
#print(mean_, std_)
text += '$S_{trans}/R=%.1f\pm%.1f$\n' %(mean_, 2*std_)

# =============================================================================
# Rotational
# =============================================================================

ax = fig.add_subplot(2,2,2)

n, _, _ = ax.hist(Srot + Srot_base, bins=60, density=True, linewidth=1, alpha=0.6, histtype='stepfilled')
ax.plot([Srot_base, Srot_base], [0, 3499], 'r--', linewidth=1.4)
ax.text(2.6723, 1500, '  Free\nRotator', fontsize=5.5, color='r')
ax.set_xlabel(r'$S_{rot}/R$', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
#ax.set_yticks([0, 0.4, 0.8, 1.2, 1.6])

ax.set_xticks([2.6710, 2.6725])
ax.set_yticks([0, 500, 1000, 1500])
#ax.set_xlim([3.8, 6])
ax.set_ylim([0, 1740])
ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.yaxis.set_label_coords(-0.2, 0.5)

ax.text(-0.18, 0.98, '(B)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)

mean_ = np.mean(Srot + Srot_base)
std_ = np.std(Srot + Srot_base)
text += '$S_{rot}/R=%.1f\pm%.1f$\n' %(mean_, 2*std_)


# =============================================================================
# Vibrational Entropy
# =============================================================================
ax = fig.add_subplot(2,2,3)

n, _, _ = ax.hist(Svib, bins=60, density=True, linewidth=1, alpha=0.6, histtype='stepfilled')
#ax.plot([Srot_base, Srot_base], [0, 58], 'r--')
#ax.text(2.657, 50, '  Free\nRotator', fontsize=6, color='r')
ax.set_xlabel(r'$S_{vib}/R$', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
ax.set_yticks([0, 0.2, 0.4, 0.6])

#ax.set_xlim([3.8, 6])
#ax.set_ylim([0, 58])
ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.yaxis.set_label_coords(-0.2, 0.5)

ax.text(-0.18, 0.98, '(C)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)

#plt.subplots_adjust(top=0.95,
#                    bottom=0.18,
#                    left=0.18,
#                    right=0.95,
#                    hspace=0.2,
#                    wspace=0.2)

mean_ = np.mean(Svib)
std_ = np.std(Svib)
text += '$S_{vib}/R=%.1f\pm%.1f$\n' %(mean_, 2*std_)
#fig.savefig('entroVib.pdf')


# =============================================================================
# Total Entropy
# =============================================================================

l = len(Svib)

Sexp = 14.2

Stot = S2D[:l] + Srot[:l] + Svib + Sconfig + Strans_base + Srot_base
np.save('Stot.npy', Stot)
mean_ = np.mean(Stot)
std_ = np.std(Stot)
text += '$S_{tot}/R=%.1f\pm%.1f$' %(mean_, 2*std_)


#fig = plt.figure(figsize=(1.5, 1.5), dpi=300)
#ax = fig.gca()
ax = fig.add_subplot(2,2,4)

m, _, _ = ax.hist(Stot, bins=50, density=True, alpha=0.6, histtype='stepfilled')
ax.plot([Sexp, Sexp], [0, 0.68], 'r--', linewidth=1.2)
ax.text(13.05, 0.45, '$S_{exp}/R$\n=%.1f\n'%Sexp, color='r', fontsize=5)
#plt.plot([Stot.mean(), Stot.mean()], [0, 1.1*max(m)], 'r:', linewidth=3, label='Mean')

ax.text(15.8, 0.42, text, fontsize=5)

ax.tick_params(axis='both', which='minor', direction='in', labelsize=5)
ax.tick_params(axis='both', which='major', direction='in', labelsize=5)
ax.set_yticks([0, 0.2, 0.4, 0.6])

ax.set_xlabel(r'$S_{tot}/R$', fontsize=7)
ax.set_ylabel('Probability Density', fontsize=7)
ax.set_xlim([13, 19])
ax.set_ylim([0, 0.68])

ax.yaxis.set_label_coords(-0.2, 0.5)

ax.text(-0.18, 0.99, '(D)', fontsize=7,
        fontweight='semibold', transform=ax.transAxes)

#plt.subplots_adjust(top=0.95,
#                    bottom=0.2,
#                    left=0.2,
#                    right=0.95,
#                    hspace=0.2,
#                    wspace=0.2)

fig.subplots_adjust(top=0.969,
                    bottom=0.135,
                    left=0.123,
                    right=0.98,
                    hspace=0.255,
                    wspace=0.345)

#plt.show()
fig.savefig('entropy_ethane.png')


#fig.savefig('Entropy_sum.pdf')
print('EHANE @106K:')
print('Mean of total entropy = %.1f +\- %.1f' %(Stot.mean(), Stot.std()))


