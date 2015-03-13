from __future__ import division, print_function, absolute_import
from os.path import join, dirname
from numpy import load, unique, zeros_like, zeros, pi, exp, float64
from scipy.interpolate import interp1d
from disvis.atompar import parameters
from saxstools.libsaxstools import scattering_curve as _scattering_curve

FORM_FACTORS_2BODY = load(join(dirname(__file__), 'data', 'form_factors_2body.npy'))
FORM_FACTORS_1BODY = load(join(dirname(__file__), 'data', 'form_factors_1body.npy'))


def scattering_curve(q, elements, xyz, out=None, bpr=2):

    # Scattering curve is given by
    # I(q) = SUM(fi*fi) + 2*SUM(SUM(fi*fj*sin(q*rij)/(q*rij)))

    if out is None:
        out = zeros_like(q, dtype=float64)

    fifj, ind1, ind2 = create_fifj_lookup_table(q, elements, elements, bpr=bpr)

    _scattering_curve(q, ind1, xyz, fifj, out)

    return out


def create_fifj_lookup_table(q, elements1, elements2, bpr=2):

    u_elements1, ind1 = unique(elements1, return_inverse=True)
    u_elements2, ind2 = unique(elements2, return_inverse=True)

    fifj = zeros((u_elements1.size, u_elements2.size, q.size), dtype=float64)
    for i, element1 in enumerate(u_elements1):

        tot_sf1 = scattering_factor(element1, q, bpr=bpr)

        for j, element2 in enumerate(u_elements2):

            fifj[i, j, :] = tot_sf1 * scattering_factor(element2, q, bpr=bpr)

    return fifj, ind1, ind2


def scattering_factor(e, q, bpr=2):
    """Returns the scattering amplitudes over a qrange"""

    if len(e) in (2, 3):
        sf = beads_scattering_factor(q, e, bpr)
    elif len(e) == 1:
        sf = element_scattering_factor(q, e)
    else:
        raise ValueError('Entity is not recognized.')

    return sf


def element_scattering_factor(q, e):

    p = parameters
    sf = element_form_factor(q, e) - solv_scattering_factor(q, p[e]['rvdW'])

    return sf


def element_form_factor(q, e):

    p = parameters
    sf = zeros(q.size, dtype=float64)
    #tmp = (q/(4*pi))**2
    #sf = p[e]['a1']*exp(-p[e]['b1']*tmp) +\
    #        p[e]['a2']*exp(-p[e]['b2']*tmp) +\
    #        p[e]['a3']*exp(-p[e]['b3']*tmp) +\
    #        p[e]['a4']*exp(-p[e]['b4']*tmp) +\
    #        p[e]['c']

    cutoff = 0.5
    ind_small_angle = q <= cutoff
    ind_high_angle = q > cutoff

    tmp = (q[ind_small_angle]/(4*pi))**2
    sf[ind_small_angle] = p[e]['a1']*exp(-p[e]['b1']*tmp) +\
            p[e]['a2']*exp(-p[e]['b2']*tmp) +\
            p[e]['a3']*exp(-p[e]['b3']*tmp) +\
            p[e]['a4']*exp(-p[e]['b4']*tmp) +\
            p[e]['c']

    highq = q[ind_high_angle]
    sf[ind_high_angle] = exp(p[e]['ha0']) *\
            exp(p[e]['ha1'] * highq) *\
            exp(p[e]['ha2'] * highq**2) *\
            exp(p[e]['ha3'] * highq**3)

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

    vvdw = (4/3)*pi*rvdw**3
    solv_sf = rho * vvdw * exp(-q**2 * vvdw**(2/3) / (4*pi))

    return solv_sf


def beads_scattering_factor(q, entity, bpr):
    
    if bpr == 1:
        data = FORM_FACTORS_1BODY
    elif bpr == 2:
        data = FORM_FACTORS_2BODY
    else:
        raise ValueError('Only 1 or 2 beads per residue are available.')

    # form factor function
    fff = interp1d(data['q'], data[entity], 
            copy=False, kind='cubic')

    return fff(q)
