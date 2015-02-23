from __future__ import division, print_function, absolute_import
import numpy as np
from .atompar import parameters
from .libsaxstools import cross_term

def saxs_curve(elements, coordinates, q, out=None):
    """Calculates a SAXS scattering curve

    Parameters
    ----------
    elements : array-like of characters
        Scattering chemical element 

    coordinates : array-like
        Nx3 array containing the xyz-coordinates of each particle

    q : array-like
        Momemtum transfer values for which the intensity will be calculated

    Returns
    -------
    out : array
        SAXS scattering intensity
    """

    if out is None:
        out = np.zeros(q.size, dtype=np.float64)

    # Scattering curve is given by
    # I(q) = SUM(fi*fi) + 2*SUM(SUM(fi*fj*sin(q*rij)/rij))
    first_term(q, elements, coordinates, out)
    second_term(q, elements, coordinates, out)

    return out


def first_term(q, elements, coordinates, out=None):

    if out is None:
        out = np.zeros_like(q)

    # calculate the first squared term taking into account the solvent displacement
    unique_elements = np.unique(elements)
    for element in unique_elements:

        sf = scattering_factor(element, q)
        solv_sf = solv_scattering_factor(q, parameters[element]['rvdW'])
        tot_sf = sf - solv_sf

        n = (np.asarray(elements) == element).sum()
        out += n * (tot_sf)**2.0

    return out


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


def cross_scattering(q, elements1, xyz1, elements2, xyz2, out=None, dfifj_table=None):

    if out is None:
        out = np.zeros_like(q)
    
    if dfifj_table is None:
        dfifj, ind1, ind2 = dfifj_lookup_table(q, elements1, elements2)
    else:
        ind1 = elements1
        ind2 = elements2
        dfifj, ind1, ind2 = dfifj_table
    
    _cross_scattering(q, ind1, xyz1, ind2, xyz2, out)

    return out


def second_term(q, elements, coordinates, out=None):

    unique_elements = list(set(elements))

    # translate elements to a number for fifj lookup
    ind_element = np.zeros(len(elements), dtype=np.int32)
    for i, element in enumerate(unique_elements):
        for j, e in enumerate(elements):
            if element == e:
                ind_element[j] = i

    # precalculate cross-terms
    n = len(unique_elements)
    fifj = np.zeros((n, n, q.size), dtype=np.float64)
    for i, element in enumerate(unique_elements):

        sf = scattering_factor(element, q)
        solv_sf = solv_scattering_factor(q, parameters[element]['rvdW'])
        tot_sf = sf - solv_sf

        for j, element2 in enumerate(unique_elements):

            sf2 = scattering_factor(element2, q)
            solv_sf2 = solv_scattering_factor(q, parameters[element2]['rvdW'])
            tot_sf2 = sf - solv_sf

            fifj[i, j, :] = tot_sf*tot_sf2

    # the cross-terms fi*fj have been precalculated.
    # now calculate the second term
    second_term = np.zeros_like(q)
    cross_term(ind_element, np.asarray(coordinates, dtype=np.float64), fifj, q, second_term)
    second_term *= 2

    if out is None:
        return second_term
    else:
        out += second_term



def scattering_factor(e, q):
    """Returns the scattering amplitudes over a qrange"""

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
