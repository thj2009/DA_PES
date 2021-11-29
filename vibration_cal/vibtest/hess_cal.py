# Read hessian

import numpy as np

# Constant
c = 299792458
NA = 6.02e23
#Unit Convertor
eV_2_J = 1.60218e-19
A_2_m = 1e-10
u_2_kg = 1.66054e-27

def read_hess(filename):
    fp = open(filename, 'r')
    
    Hess = []
    for idx, line in enumerate(fp.readlines()):
        line = line.strip()
        if idx == 2:
            nentry = len(line.split())
        if idx >= 3:
            # start reading the hessian
            sline = line.split()
            row = [float(term) for term in sline[1:]]
            Hess.append(row)
    fp.close()
    
    Hess = np.array(Hess)
    assert np.shape(Hess) == (nentry, nentry)
    # Symmtreized
    Hess = -1/2. * (Hess + Hess.T)
    return Hess

def vibcal(Hess, mass):
    mass = np.repeat(mass, 3)
    inv_mass = np.diag(1/mass)
    # mass weighted hess
    weight_H = np.sqrt(inv_mass).dot(Hess).dot(np.sqrt(inv_mass))
    eig_ = np.linalg.eigvals(weight_H)
    # sorted out
    eig_ = np.sort(eig_)
    # Convert unit
    eig_ = eig_ * eV_2_J / (A_2_m**2 * u_2_kg)
    # eigenvalue less than zero repalce with zero
    eig_ = [max(e, 0) for e in eig_]
    freq = np.sqrt(eig_)/(2*np.pi)
    wave_len = freq/c/100
    return freq, wave_len
