from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cimport cython

@cython.boundscheck(False) # turn of bounds-checking for entire function
def middle_values(np.ndarray[DTYPE_t, ndim=2] f,
                           np.ndarray[DTYPE_t, ndim=2] g):
    assert f.dtype == DTYPE and g.dtype == DTYPE
    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2*smid
    cdef int ymax = wmax + 2*tmid
    cdef np.ndarray[DTYPE_t, ndim=3] h = np.zeros([vmax, smax, wmax], dtype=DTYPE)
    cdef int s, t, v, w, first_index, second_index, third_index
    cdef int s_from, s_to, t_from, t_to
    cdef DTYPE_t value
    for first_index in range(vmax):
        for second_index in range(smax):
            for third_index in range(wmax):
                h[first_index, second_index, third_index] = (f[first_index, third_index] + g[second_index, third_index]) / 2
    return h
