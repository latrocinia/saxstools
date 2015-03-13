import cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scattering_curve(np.ndarray[np.float64_t, ndim=1] q,
                     np.ndarray[np.int64_t, ndim=1] ind,
                     np.ndarray[np.float64_t, ndim=2] xyz,
                     np.ndarray[np.float64_t, ndim=3] fifj,
                     np.ndarray[np.float64_t, ndim=1] out,
                     ):

    cdef unsigned int i, j, k
    cdef unsigned int nq = q.shape[0]
    cdef unsigned int npoints = ind.shape[0]
    cdef double rij, qrij

    # first term
    for i in range(nq):
        for j in range(npoints):
            out[i] += fifj[ind[j], ind[j], i]
    
    # second expensive term
    for i in range(1, npoints - 1):
        for j in range(i + 1, npoints):
            rij = sqrt((xyz[i, 0] - xyz[j, 0])**2 + (xyz[i, 1] - xyz[j, 1])**2 + (xyz[i, 2] - xyz[j, 2])**2)
            for k in range(nq):
                qrij = q[k] * rij
                if qrij == 0:
                    out[k] += 2 * fifj[ind[i], ind[j], k]
                else:
                    out[k] += 2 * fifj[ind[i], ind[j], k] * sin(qrij)/qrij
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void cross_term(np.ndarray[np.float64_t, ndim=1] q,
               np.ndarray[np.int64_t, ndim=1] ind1,
               np.ndarray[np.float64_t, ndim=2] xyz1,
               np.ndarray[np.int64_t, ndim=1] ind2,
               np.ndarray[np.float64_t, ndim=2] xyz2,
               np.ndarray[np.float64_t, ndim=3] fifj,
               np.ndarray[np.float64_t, ndim=1] out,
               ):
    
    cdef unsigned int i, j, k
    cdef double rij, qrij

    for i in range(ind1.shape[0]):
        for j in range(ind2.shape[0]):
            rij = sqrt((xyz1[i, 0] - xyz2[j, 0])**2 + (xyz1[i, 1] - xyz2[j, 1])**2 + (xyz1[i, 2] - xyz2[j, 2])**2)

            for k in range(q.shape[0]):
                qrij = q[k] * rij
                if qrij == 0:
                    out[k] += 2 * fifj[ind1[i], ind2[j], k]
                else:
                    out[k] += 2 * fifj[ind1[i], ind2[j], k] * sin(qrij)/qrij

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_chi(np.ndarray[np.int32_t, ndim=3] interspace,
             np.ndarray[np.float64_t, ndim=1] q,
             np.ndarray[np.float64_t, ndim=1] Iq,
             np.ndarray[np.int64_t, ndim=1] rind,
             np.ndarray[np.float64_t, ndim=2] rxyz,
             np.ndarray[np.int64_t, ndim=1] lind,
             np.ndarray[np.float64_t, ndim=2] lxyz,
             double origin,
             double voxelspacing,
             np.ndarray[np.float64_t, ndim=3] fifj,
             np.ndarray[np.float64_t, ndim=1] targetIq,
             np.ndarray[np.float64_t, ndim=3] chi2,
             ):

    cdef unsigned x, y, z, n
    cdef double sumIqtargetIq, sumIq2, fscale

    cdef np.ndarray[np.float64_t, ndim=2] tmplxyz = np.zeros_like(lxyz)
    cdef np.ndarray[np.float64_t, ndim=1] tmpIq = np.zeros(Iq.shape[0], dtype=np.float64)

    for z in range(chi2.shape[0]):
        for y in range(chi2.shape[1]):
            for x in range(chi2.shape[2]):

                if interspace[z, y, x] == 0:
                    continue

                # move the coordinates of the ligand
                for n in range(lxyz.shape[0]):
                    tmplxyz[n, 0] = lxyz[n, 0] + x*voxelspacing + origin
                    tmplxyz[n, 1] = lxyz[n, 1] + y*voxelspacing + origin
                    tmplxyz[n, 2] = lxyz[n, 2] + z*voxelspacing + origin

                # calculate the cross scattering
                for n in range(q.shape[0]):
                    tmpIq[n] = Iq[n]
                cross_term(q, rind, rxyz, lind, tmplxyz, fifj, tmpIq)

                # scale the calculated Iq
                sumIqtargetIq = 0
                sumIq2 = 0
                for n in range(q.shape[0]):
                    sumIqtargetIq += tmpIq[n] * targetIq[n]
                    sumIq2 += tmpIq[n]*tmpIq[n]
                fscale = sumIqtargetIq/sumIq2

                # calculate chi2
                for n in range(q.shape[0]):
                    chi2[z, y, x] += (targetIq[n] - tmpIq[n])**2
                chi2[z, y, x] /= q.shape[0]
#
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
#def cross_term(np.ndarray[np.int32_t, ndim=1] ie,
#               np.ndarray[np.float64_t, ndim=2] coor,
#               np.ndarray[np.float64_t, ndim=3] fifj2,
#               np.ndarray[np.float64_t, ndim=1] q,
#               np.ndarray[np.float64_t, ndim=1] out):
#    """Calculates the cross term"""
#
#    cdef unsigned int n, m, i, j, qn
#    cdef double qrij, rij
#
#    n = <int> len(ie)
#    qn = <int> q.size
#
#    for i in range(n - 1):
#        for j in range(i+1, n):
#
#            rij = sqrt((coor[i, 0] - coor[j,0])**2 +\
#                  (coor[i,1] - coor[j,1])**2 +\
#                  (coor[i,2] - coor[j,2])**2)
#
#            for m in range(qn):
#
#                qrij = q[m]*rij
#                if qrij == 0:
#                    out[m] += fifj2[ie[i], ie[j], m]
#                else:
#                    out[m] += fifj2[ie[i], ie[j], m]*sin(qrij)/(qrij)
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#@cython.cdivision(True)
#def cross_scattering(np.ndarray[np.float64_t, ndim=1] q,
#                     np.ndarray[np.int32_t, ndim=1] ind1,
#                     np.ndarray[np.float64_t, ndim=2] xyz1,
#                     np.ndarray[np.int32_t, ndim=1] ind2,
#                     np.ndarray[np.float64_t, ndim=2] xyz2,
#                     np.ndarray[np.float64_t, ndim=3] dfifj,
#                     np.ndarray[np.float64_t, ndim=1] out):
#    """Calculates the scattering between two sets"""
#
#    cdef unsigned int n, i, j
#    cdef double qrij, rij
#
#    for i in range(ind1.size):
#        for j in range(ind2.size):
#
#            rij = sqrt((xyz1[i, 0] - xyz2[j, 0])**2 +\
#                  (xyz1[i,1] - xyz2[j, 1])**2 +\
#                  (xyz1[i,2] - xyz2[j, 2])**2)
#
#            for n in range(q.size):
#                qrij = q[n]*rij
#                if qrij == 0:
#                    out[n] += dfifj[ind1[i], ind2[j], n]
#                else:
#                    out[n] += dfifj[ind1[i], ind2[j], n]*sin(qrij)/(qrij)
#
