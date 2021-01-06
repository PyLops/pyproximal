import logging
import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import SimplexProj

try:
    from numba import jit
    from ._Simplex_numba import bisect_jit, simplex_jit, fun_jit
except ModuleNotFoundError:
    jit = None
    jit_message = 'Numba not available, reverting to numpy.'
except Exception as e:
    jit = None
    jit_message = 'Failed to import numba (error:%s), use numpy.' % e

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.WARNING)


class _Simplex(ProxOperator):
    """Simplex operator (numpy version)
    """
    def __init__(self, n, radius, dims=None, axis=-1, maxiter=100, xtol=1e-8,
                 call=True):
        super().__init__(None, False)
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n = n
        self.radius = radius
        self.dims = dims
        self.axis = axis
        self.otheraxis = 1 if axis == 0 else 0
        self.maxiter = maxiter
        self.xtol = xtol
        self.call = call

        self.simplex = SimplexProj(self.n if dims is None else dims[axis],
                                   self.radius, maxiter=self.maxiter,
                                   xtol=self.xtol)

    def __call__(self, x, tol=1e-8):
        if not self.call:
            return False
        if self.dims is None:
            radcheck = np.sum(x) - self.radius > tol or \
                       np.sum(x) - self.radius < -tol
            c = not (radcheck or (np.any(x < 0)))
        else:
            x = x.reshape(self.dims)
            if self.axis == 0:
                x = x.T
            c = np.zeros(self.dims[self.otheraxis], dtype=np.bool)
            for i in range(self.dims[self.otheraxis]):
                c[i] = not (np.abs(np.sum(x)) - self.radius < tol or np.any(x[i] < 0))
            c = np.all(c)
        return c

    @_check_tau
    def prox(self, x, tau):
        if self.dims is None:
            y = self.simplex(x)
        else:
            x = x.reshape(self.dims)
            if self.axis == 0:
                x = x.T
            y = np.zeros_like(x)
            for i in range(self.dims[self.otheraxis]):
                y[i] = self.simplex(x[i])
            if self.axis == 0:
                y = y.T
        return y.ravel()


class _Simplex_numba(_Simplex):
    """Simplex operator (numba version)
   """
    def __init__(self, n, radius, dims=None, axis=-1,
                 maxiter=100, ftol=1e-8, xtol=1e-8, call=False):
        super().__init__(None, False)
        if dims is not None and len(dims) != 2:
            raise ValueError('provide only 2 dimensions, or None')
        self.n = n
        self.coeffs = np.ones(self.n if dims is None else dims[axis])
        self.radius = radius
        self.dims = dims
        self.axis = axis
        self.otheraxis = 1 if axis == 0 else 0
        self.maxiter = maxiter
        self.ftol = ftol
        self.xtol = xtol
        self.call = call

    def prox(self, x, tau):
        if self.dims is None:
            bisect_lower = -1
            while fun_jit(bisect_lower, x, self.coeffs, self.radius, 0, 10000000000) < 0:
                bisect_lower *= 2
            bisect_upper = 1
            while fun_jit(bisect_upper, x, self.coeffs, self.radius, 0, 10000000000) > 0:
                bisect_upper *= 2
            c = bisect_jit(x, self.coeffs, self.radius, 0, 10000000000,
                           bisect_lower, bisect_upper, self.maxiter,
                           self.ftol, self.xtol)
            y = np.minimum(np.maximum(x - c * self.coeffs, 0), 10000000000)
        else:
            x = x.reshape(self.dims)
            if self.axis == 0:
                x = x.T
            y = simplex_jit(x, self.coeffs, self.radius, 0, 10000000000,
                            self.maxiter, self.ftol, self.xtol)
            if self.axis == 0:
                y = y.T
        return y.ravel()


def Simplex(n, radius, dims=None, axis=-1, maxiter=100,
            ftol=1e-8, xtol=1e-8, call=True, engine='numpy'):
    r"""Simplex proximal operator.

    Proximal operator of a Simplex: :math:`\Delta_n(r) = \{ \mathbf{x}:
    \sum_i x_i = r,\; x_i \geq 0 \}`. This operator can be applied to a
    single vector as well as repeatedly to a set of vectors which are
    defined as the rows (or columns) of a matrix obtained by reshaping the
    input vector as defined by the ``dims`` and ``axis`` parameters.

    Parameters
    ----------
    n : :obj:`int`
        Number of elements of input vector
    radius : :obj:`float`
        Radius
    dims : :obj:`tuple`, optional
        Dimensions of the matrix onto which the input vector is reshaped
    axis : :obj:`int`, optional
        Axis along which simplex is repeatedly applied when ``dims`` is not
        provided
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by bisection
    ftol : :obj:`float`, optional
        Function tolerance in bisection (only with ``engine='numba'``)
    xtol : :obj:`float`, optional
        Solution absolute tolerance in bisection
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)
    engine : :obj:`str`, optional
        Engine used for simplex computation (``numpy`` or ``numba``).

    Raises
    ------
    KeyError
        If ``engine`` is neither ``numpy`` nor ``numba``
    ValueError
        If ``dims`` is provided as a list (or tuple) with more or less than
        2 elements

    Notes
    -----
    As the Simplex is an indicator function, the proximal operator corresponds
    to its orthogonal projection (see :class:`pyprox.projection.SimplexProj`
    for details.

    Note that ``tau`` does not have effect for this proximal operator, any
    positive number can be provided.

    """
    if not engine in ['numpy', 'numba']:
        raise KeyError('engine must be numpy or numba')

    if engine == 'numba' and jit is not None:
        s = _Simplex_numba(n, radius, dims=dims, axis=axis,
                           maxiter=maxiter, ftol=ftol, xtol=xtol, call=call)
    else:
        if engine == 'numba' and jit is None:
            logging.warning(jit_message)
        s = _Simplex(n, radius, dims=dims, axis=axis,
                     maxiter=maxiter, xtol=xtol, call=call)
    return s