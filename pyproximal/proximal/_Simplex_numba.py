import os
import numpy as np
from numba import jit, prange

# detect whether to use parallel or not
numba_threads = int(os.getenv('NUMBA_NUM_THREADS', '1'))
parallel = True if numba_threads != 1 else False


@jit(nopython=True)
def fun_jit(mu, x, coeffs, scalar, lower, upper):
    """Bisection function"""
    return np.dot(coeffs, np.minimum(np.maximum(x - mu * coeffs, lower), upper)) - scalar


@jit(nopython=True, nogil=True)
def bisect_jit(x, coeffs, scalar, lower, upper, bisect_lower, bisect_upper,
               maxiter, ftol, xtol):
    """Bisection method

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Input vector
    coeffs : :obj:`np.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound of Box
    bisect_lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower end of bisection
    bisect_upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper end of bisection
    maxiter : :obj:`int`, optional
        Maximum number of iterations
    ftol : :obj:`float`, optional
        Function tolerance
    xtol : :obj:`float`, optional
        Solution absolute tolerance

    """
    a, b = bisect_lower, bisect_upper
    fa = fun_jit(a, x, coeffs, scalar, lower, upper)
    for iiter in range(maxiter):
        c = (a + b) / 2.
        if (b - a)/2 < xtol:
            return c
        fc = fun_jit(c, x, coeffs, scalar, lower, upper)
        if np.abs(fc) < ftol:
            return c
        if np.sign(fc) == np.sign(fa):
            a = c
            fa = fc
        else:
            b = c
    return c


@jit(nopython=True, parallel=parallel, nogil=True)
def simplex_jit(x, coeffs, scalar, lower, upper, maxiter, ftol, xtol):
    """Simplex proximal

    Parameters
    ----------
    x : :obj:`np.ndarray`
        Input vector
    coeffs : :obj:`np.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound of Box
    maxiter : :obj:`int`, optional
        Maximum number of iterations
    ftol : :obj:`float`, optional
        Function tolerance
    xtol : :obj:`float`, optional
        Solution absolute tolerance

    """
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        bisect_lower = -1
        while fun_jit(bisect_lower, x[i], coeffs, scalar, lower, upper) < 0:
            bisect_lower *= 2
        bisect_upper = 1
        while fun_jit(bisect_upper, x[i], coeffs, scalar, lower, upper) > 0:
            bisect_upper *= 2
        c = bisect_jit(x[i], coeffs, scalar, lower, upper,
                       bisect_lower, bisect_upper, maxiter, ftol, xtol)
        y[i] = np.minimum(np.maximum(x[i] - c * coeffs, lower), upper)
    return y
