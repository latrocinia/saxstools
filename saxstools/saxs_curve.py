from __future__ import division, print_function, absolute_import
import numpy as np
from .atompar import parameters
from .libsaxstools import cross_term

def saxs_curve(elements, coordinates, q):
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
    Iq : array
        SAXS scattering intensity
    """

    Iq = np.zeros(q.size, dtype=np.float64)

    # Scattering curve is given by
    # I(q) = SUM(fi*fi) + 2*SUM(SUM(fi*fj*sin(q*rij)/rij))
    first_term(elements, coordinates, q, Iq)
    second_term(elements, coordinates, q, Iq)

    return Iq

def first_term(elements, coordinates, q, out=None):

    return_out = False
    if out is None:
        out = np.zeros_like(q)
        return_out = True

    # calculate the first squared term taking into account the solvent displacement
    unique_elements = list(set(elements))
    n = len(unique_elements)
    for element in unique_elements:

        sf = scattering_factor(element, q)
        solv_sf = solv_scattering_factor(q, rvdw=parameters[element]['rvdW'])
        tot_sf = sf - solv_sf

        n = elements.count(element)
        out += n*(tot_sf)**2.0

    if return_out:
        return out

def second_term(elements, coordinates, q, out=None):

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
        solv_sf = solv_scattering_factor(q, rvdw=parameters[element]['rvdW'])
        tot_sf = sf - solv_sf

        for j, element2 in enumerate(unique_elements):

            sf2 = scattering_factor(element2, q)
            solv_sf2 = solv_scattering_factor(q, rvdw=parameters[element2]['rvdW'])
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
    tmp = (q/(4*np.pi))**2
    sf = p[e]['a1']*np.exp(-p[e]['b1']*tmp) +\
         p[e]['a2']*np.exp(-p[e]['b2']*tmp) +\
         p[e]['a3']*np.exp(-p[e]['b3']*tmp) +\
         p[e]['a4']*np.exp(-p[e]['b4']*tmp) +\
         p[e]['c']

    return sf

def solv_scattering_factor(q, rvdw=None, rho=0.334):
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
