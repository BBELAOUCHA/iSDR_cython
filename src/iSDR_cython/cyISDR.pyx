# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Alexis Mignon <alexis.mignon@gmail.com>
#         Manoj Kumar <manojkumarsivaraj334@gmail.com>
#
# License: BSD 3 clause
#
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport fabs
cimport numpy as np
import numpy as np
import numpy.linalg as linalg

cimport cython
from cpython cimport bool
from cython cimport floating
import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils._cython_blas cimport (_axpy, _dot, _asum, _ger, _gemv, _nrm2, 
                                   _copy, _scal)
from sklearn.utils._cython_blas cimport RowMajor, ColMajor, Trans, NoTrans


from sklearn.utils._random cimport our_rand_r

ctypedef np.float64_t DOUBLE
ctypedef np.uint32_t UINT32_t

np.import_array()

# The following two functions are shamelessly copied from the tree code.

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) nogil:
    if x > y:
        return x
    return y


cdef inline floating fsign(floating f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


cdef floating abs_max(int n, floating* a) nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating max(int n, floating* a) nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef floating diff_abs_max(int n, floating* a, floating* b) nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m


def enet_coordinate_descent_iSDR(np.ndarray[floating, ndim=1] w,
                            floating alpha,
                            np.ndarray[floating, ndim=2, mode='fortran'] X,
                            np.ndarray[floating, ndim=1] y,
                            int m_p, int max_iter, floating tol,
                            object rng, bint random=0, int verbose=0):
    """Cython version iSDR


        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w)_21
    """
    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64
    # get the data information into easy vars
    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    # get the number of tasks indirectly, using strides
    cdef unsigned int n_tasks = y.strides[0]
    cdef unsigned int n_s = n_features / m_p
    cdef unsigned int n_c = n_samples
    cdef unsigned int n_t = len(y) / n_c
    cdef unsigned int n_t_s = n_t + m_p - 1
    cdef unsigned int block = n_s * n_c
    # compute norms of the columns of X (nbr_sources * model mar)
    cdef floating[:] norm_cols_X = (X**2).sum(axis=0)
    cdef floating[:] mu_X = np.zeros(n_s, dtype=dtype) # mu = 1/L lip
    # initial value of the residuals
    cdef floating[:] R = np.zeros(len(y), dtype=dtype)
    cdef floating[:, ::1] XtA = np.zeros((n_s, n_t_s), dtype=dtype) #
    cdef floating[::1] tmp = np.zeros(n_t_s, dtype=dtype)
    cdef floating[:] w_ii = np.zeros(n_t_s, dtype=dtype)
    cdef floating d_w_max
    cdef floating W_ii_abs_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating XtA_axis1norm
    cdef floating R_norm
    cdef floating w_norm
    cdef floating l1_norm
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int jj
    cdef unsigned int j
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed
    cdef floating* X_ptr = &X[0, 0]
    cdef floating* W_ptr = &w[0]
    cdef floating* wii_ptr = &w_ii[0]
    cdef floating* Y_ptr = &y[0]
    cdef floating* R_ptr = &R[0]
    cdef floating ry_sum # R^TY
    cdef floating nn
    cdef floating s
    cdef floating l21_norm
    
    if alpha == 0:
        if verbose:
            warnings.warn("Coordinate descent with alpha=0 may lead to unexpected"
                " results and is discouraged.")
    for jj in range(n_s):
        s = 0.0
        for ii in range(m_p):
            for i in range(n_c):
                s += X[i, jj + ii*n_s]**2
        if s != 0.0:
                mu_X[jj] =  1.0 / s
    with nogil:
        # R = y - np.dot(X, w)
        _copy(n_t * n_c, Y_ptr, 1, R_ptr, 1)
        # tol *= np.dot(y, y)
        tol *= _nrm2(n_c * n_t, Y_ptr, 1) ** 2
        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_s):  # Loop over coordinates
                if random:
                    ii = rand_int(n_s, rand_r_state)
                else:
                    ii = f_iter
                # Store previous value w_ii = w[ii cd all]
                _copy(n_t_s, W_ptr + ii, n_s, wii_ptr, 1)
                _copy(n_t_s, wii_ptr, 1, &tmp[0], 1)
                # tmp = np.dot(X[:, ii][None, :], R).ravel()
                for i in range(m_p - 1):
                    # First block
                    for j in range(i + 1):
                        tmp[i] += mu_X[ii]*(_dot(n_c, X_ptr + ii * n_c + (i - j) * block, 1,
                              &R[0] + j * n_c, 1))
                    # Last block
                    jj = n_t_s + i - m_p + 1
                    for j in range(n_t_s - jj):
                        tmp[jj] += mu_X[ii]*(_dot(n_c, X_ptr + ii * n_c + (m_p - 1 - j) * block, 1, &R[0] + (n_t - m_p + 1 + j) * n_c, 1))
                # Middle block
                for i in range(m_p - 1, n_t_s - m_p + 1):
                    for j in range(m_p):
                        tmp[i] += mu_X[ii]*(_dot(n_c, X_ptr + ii*n_c + (m_p - 1 - j) * block, 1, &R[0] + (j + i + 1 - m_p) * n_c , 1))

                nn = _nrm2(n_t_s, &tmp[0], 1)
                # copy tmp back to w[ii, :]
                s = fmax(nn, mu_X[ii] * alpha)
                _scal(n_t_s, fmax(1.0 - (mu_X[ii] * alpha) / s, 0.0), &tmp[0], 1)
                _copy(n_t_s, &tmp[0], 1, W_ptr + ii, n_s)
                d_w_ii = diff_abs_max(n_t_s, &tmp[0], wii_ptr)
                W_ii_abs_max = abs_max(n_t_s, &tmp[0])
                if d_w_ii != 0.0:
                    #for jj in range(n_t): # n_t = T-p
                    #    for j in range(n_c):
                    #        for i in range(m_p):
                    #            R[j + jj * n_c] -= X[j, ii + i * n_s] * (tmp[jj + i] - w_ii[jj + i])                            
                    for jj in range(n_t_s):
                        tmp[jj] -= w_ii[jj]

                    for jj in range(n_t):
                        _gemv(ColMajor, NoTrans, n_c, m_p, -1, X_ptr + ii*n_c, n_c*n_s, &tmp[0] + jj, 1, 1, &R[0] + jj*n_c, 1)

                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii

                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max
            if w_max == 0.0 or d_w_max / w_max < d_w_tol or n_iter == max_iter - 1:
                # the biggest coordinate update of this iteration was smaller
                # than the tolerance: check the duality gap as ultimate
                # stopping criterion
                # XtA = np.dot(X.T, R)
                for ii in range(n_s):
                    for jj in range(m_p - 1, n_t_s - 1):
                        s = 0.0
                        for i in range(m_p):
                            s += (_dot(n_c, X_ptr + ii * n_c + (m_p - 1 - i) * block, 1,  &R[0] + (jj + i - m_p + 1 ) * n_c, 1))
                        XtA[ii, jj] = s
    #
                    for i in range(m_p - 1):
                        s = 0.0
                        for j in range(0, i + 1):
                            s += (_dot(n_c, X_ptr + ii * n_c + (i-j) * block, 1, &R[0] + j * n_c, 1))
                        XtA[ii, i] = s
                    XtA[ii, n_t_s - 1] = (_dot(n_c, X_ptr + ii * n_c + (m_p - 1 ) * block, 1, &R[0] + (n_t - 1) * n_c, 1))
    
                # dual_norm_XtA = np.max(np.sqrt(np.sum(XtA ** 2, axis=1)))
                dual_norm_XtA = 0.0
                for ii in range(n_s):
                    # np.sqrt(np.sum(XtA ** 2, axis=1))
                    XtA_axis1norm = _nrm2(n_t_s, &XtA[0, 0] + ii * n_t_s, 1)
                    if XtA_axis1norm > dual_norm_XtA:
                        dual_norm_XtA = XtA_axis1norm
    
                R_norm = _nrm2(n_t * n_c, &R[0], 1)
                w_norm = _nrm2(n_s * n_t_s, W_ptr, 1)
    
                if (dual_norm_XtA > alpha):
                    const =  alpha / dual_norm_XtA
                    A_norm = R_norm * const
                    gap = 0.5 * (R_norm ** 2 + A_norm ** 2)
                else:
                    const = 1.0
                    gap = R_norm ** 2
    
                # ry_sum = np.sum(R * y)
                ry_sum = 0.0
                for ii in range(n_t*n_c):
                        ry_sum += R[ii] * y[ii]
                # l21_norm = np.sqrt(np.sum(W ** 2, axis=0)).sum()
                l21_norm = 0.0
    
                for ii in range(n_s):
                    # np.sqrt(np.sum(W ** 2, axis=0))
                    l21_norm += _nrm2(n_t_s, W_ptr + ii, n_s)
                gap += alpha * l21_norm - const * ry_sum
    
                if gap < tol:
                    # return if we reached desired tolerance
                    break
        else:
                # for/else, runs if for doesn't end with a `break`
            with gil:
                if verbose:
                    warnings.warn("Objective did not converge. You might want to "
                                  "increase the number of iterations. Duality "
                                  "gap: {}, tolerance: {}".format(gap, tol),
                                  ConvergenceWarning)
    return np.asarray(w), gap, tol, n_iter + 1
   
