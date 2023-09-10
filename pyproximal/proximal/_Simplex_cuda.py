from numba import cuda


@cuda.jit(device=True)
def fun_jit_cuda(mu, x, coeffs, scalar, lower, upper):
    """Bisection function"""
    p = 0
    for i in range(coeffs.shape[0]):
        p += coeffs[i] * min(max(x[i] - mu * coeffs[i], lower), upper)
    return p - scalar


@cuda.jit(device=True)
def bisect_jit_cuda(x, coeffs, scalar, lower, upper, bisect_lower, bisect_upper,
                    maxiter, ftol, xtol):
    """Bisection method (See _Simplex_numba for details).

    """
    a, b = bisect_lower, bisect_upper
    fa = fun_jit_cuda(a, x, coeffs, scalar, lower, upper)
    for iiter in range(maxiter):
        c = (a + b) / 2.
        if (b - a) / 2 < xtol:
            return c
        fc = fun_jit_cuda(c, x, coeffs, scalar, lower, upper)
        if abs(fc) < ftol:
            return c
        if fc / abs(fc) == fa / abs(fa):
            a = c
            fa = fc
        else:
            b = c
    return c


@cuda.jit
def simplex_jit_cuda(x, coeffs, scalar, lower, upper, maxiter, ftol, xtol, y):
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
    y : :obj:`np.ndarray`
        Output vector

    """
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if i < x.shape[0]:
        bisect_lower = -1
        while fun_jit_cuda(bisect_lower, x[i], coeffs, scalar, lower, upper) < 0:
            bisect_lower *= 2
        bisect_upper = 1
        while fun_jit_cuda(bisect_upper, x[i], coeffs, scalar, lower, upper) > 0:
            bisect_upper *= 2

        c = bisect_jit_cuda(x[i], coeffs, scalar, lower, upper,
                            bisect_lower, bisect_upper, maxiter, ftol, xtol)

        for j in range(coeffs.shape[0]):
            y[i][j] = min(max(x[i][j] - c * coeffs[j], lower), upper)