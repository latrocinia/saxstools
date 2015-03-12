from __future__ import division, print_function, absolute_import
import os.path
import numpy as np
from scipy.interpolate import interp1d
from .atompar import parameters

FORM_FACTORS_2BODY = np.load(os.path.join(os.path.dirname(__file__), 'data', 'form_factors_2body.npy'))

    # Scattering curve is given by
    # I(q) = SUM(fi*fi) + 2*SUM(SUM(fi*fj*sin(q*rij)/rij))


def dfifj_lookup_table(q, elements1, elements2):

    u_elements1, ind1 = np.unique(elements1, return_inverse=True)
    u_elements2, ind2 = np.unique(elements2, return_inverse=True)

    ni = u_elements1.size
    nj = u_elements2.size
    dfifj = np.zeros((ni, nj, q.size), dtype=np.float64)
    for i, element1 in enumerate(u_elements1):

        tot_sf1 = scattering_factor(element1, q) - \
                solv_scattering_factor(q, parameters[element1]['rvdW'])

        for j, element2 in enumerate(u_elements2):

            dfifj[i, j, :] = 2 * tot_sf1 * (scattering_factor(element2, q) -\
                    solv_scattering_factor(q, parameters[element2]['rvdW']))

    return dfifj, ind1, ind2


def scattering_factor(e, q):
    """Returns the scattering amplitudes over a qrange"""

    if len(e) == 3:
        sf = form_factor_2body(q, e)

    elif len(e) == 1:
        p = parameters
        sf = np.zeros(q.size, dtype=np.float64)

        ind_small_angle = q < 2.0
        ind_high_angle = q >= 2.0

        tmp = (q[ind_small_angle]/(4*np.pi))**2
        sf[ind_small_angle] = p[e]['a1']*np.exp(-p[e]['b1']*tmp) +\
                p[e]['a2']*np.exp(-p[e]['b2']*tmp) +\
                p[e]['a3']*np.exp(-p[e]['b3']*tmp) +\
                p[e]['a4']*np.exp(-p[e]['b4']*tmp) +\
                p[e]['c']

        sf[ind_high_angle] = np.exp(p[e]['ha0']) *\
                np.exp(p[e]['ha1'] * q) *\
                np.exp(p[e]['ha2'] * q**2) *\
                np.exp(p[e]['ha3'] * q**3)

        sf -= solv_scattering_factor(q, p[e]['rvdW'])
    else:
        raise ValueError('Entity is not recognized.')

    return sf


def solv_scattering_factor(q, rvdw, rho=0.334):
    """Returns the scattering factor of displaced solvent

    Parameters
    ----------
    q : float or array
        Momentum transfer in A**-1
    rvdw : float
        Van der Waals radius of displacing element in A
    rho : float
        Average electron density of solvent in A**3
    
    Returns
    -------
    Scattering factor of displaced solvent
    """

    vvdw = (4/3)*np.pi*rvdw**3
    solv_sf = rho*vvdw*np.exp(-q**2*vvdw**(2/3)/(4*np.pi))

    return solv_sf


def form_factor_2body(q, entity):
    
    # form factor function
    fff = interp1d(FORM_FACTORS_2BODY['q'], FORM_FACTORS_2BODY[entity], 
            copy=False, kind='cubic', assume_sorted=True)

    return fff(q)
