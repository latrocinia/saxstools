import cython
import numpy as np
cimport numpy as np
from libc.math cimport sin, sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cross_term(np.ndarray[np.int32_t, ndim=1] ie, 
               np.ndarray[np.float64_t, ndim=2] coor, 
               np.ndarray[np.float64_t, ndim=3] fifj2,
               np.ndarray[np.float64_t, ndim=1] q,
               np.ndarray[np.float64_t, ndim=1] out):
    """Calculates the cross term"""

    cdef unsigned int n, m, i, j, qn
    cdef double qrij, rij

    n = <int> len(ie)
    qn = <int> q.size

    for i in range(n - 1):
        for j in range(i+1, n):

            rij = sqrt((coor[i, 0] - coor[j,0])**2 +\
                  (coor[i,1] - coor[j,1])**2 +\
                  (coor[i,2] - coor[j,2])**2)

            for m in range(qn):

                qrij = q[m]*rij
                out[m] += fifj2[ie[i], ie[j], m]*sin(qrij)/(qrij)
