from __future__ import division, print_function, absolute_import
import numpy as np
from .atompar import parameters
from .libsaxstools import cross_term

def saxs_curve(elements, coordinates, qstart=0.00, qend=0.33, steps=10):

    q = np.linspace(qstart, qend, num=steps, dtype=np.float64)
    Iq = np.zeros(q.size, dtype=np.float64)

    # Scattering curve is given by
    # I(q) = SUM(fi*fi) + 2*SUM(SUM(fi*fj*sin(q*rij)/rij))

    # calculate the first squared term taking into account the solvent displacement
    # and precalculate the cross-terms in the second term
    unique_elements = list(set(elements))
    n = len(unique_elements)
    fifj2 = np.zeros((n, n, q.size), dtype=np.float64)
    for i, element in enumerate(unique_elements):

        sf = scattering_factor(element, q)
        solv_sf = solv_scattering_factor(q, rvdw=parameters[element]['rvdW'])
        tot_sf = sf - solv_sf

        n = elements.count(element)
        Iq += n*(tot_sf)**2.0

        for j, element2 in enumerate(unique_elements):
            sf2 = scattering_factor(element2, q)
            solv_sf2 = solv_scattering_factor(q, rvdw=parameters[element2]['rvdW'])
            tot_sf2 = sf - solv_sf
            fifj2[i, j, :] = 2*tot_sf*tot_sf2

    # translate elements to a number for fifj lookup
    ind_element = np.zeros(len(elements), dtype=np.int32)
    for i, element in enumerate(unique_elements):
        for j, e in enumerate(elements):
            if element == e:
                ind_element[j] = i

    # the cross-terms fi*fj and the pairwise distances rij have
    # been precalculated.
    # now calculate the second term
    cross_term(ind_element, np.asarray(coordinates, dtype=np.float64), fifj2, q, Iq)

    return q, Iq

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
