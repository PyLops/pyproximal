import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Geman(ProxOperator):
    r"""Geman penalty.

    The Geman penalty (named after its inventor) is a non-convex penalty [1]_.
    The pyproximal implementation considers a generalized model where

    .. math::

        \mathrm{Geman}_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma |x_i|}{|x_i| + \gamma}

    where :math:`{\sigma\geq 0}`, :math:`{\gamma>0}`.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.3.

    Notes
    -----
    In order to compute the proximal operator of the Geman penalty one must find the
    roots of a cubic polynomial. Consider the one-dimensional problem

    .. math::
        \prox_{\tau \mathrm{Geman}(\cdot)}(x) = \argmin_{z} \mathrm{Geman}(z) + \frac{1}{2\tau}(x - z)^2

    and assume :math:`{x\geq 0}`. Either the minimum is obtained when :math:`z=0` or
    when

    .. math::
        \tau\sigma\gamma + (z-x)(z+\gamma)^2 = 0 .

    The pyproximal implementation uses the closed-form solution for a cubic equation,
    and discards infeasible roots, to find the minimum.

    .. [1] Geman and Yang "Nonlinear image recovery with half-quadratic regularization",
        IEEE Transactions on Image Processing, 4(7):932 – 946, 1995.

    """

    def __init__(self, sigma, gamma=1.3):
        super().__init__(None, False)
        if sigma < 0:
            raise ValueError('Variable "sigma" must be positive.')
        if gamma <= 0:
            raise ValueError('Variable "gamma" must be strictly positive.')
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.sigma * np.abs(x) / (np.abs(x) + self.gamma)

    @_check_tau
    def prox(self, x, tau):
        out = np.zeros_like(x)
        b = 2 * self.gamma - np.abs(x)
        c = self.gamma ** 2 - 2 * self.gamma * np.abs(x)
        d = self.gamma * self.sigma * tau - self.gamma ** 2 * np.abs(x)
        idx, loc_mins = self._find_local_minima(b, c, d)
        global_min_idx = tau * self.elementwise(loc_mins) + \
            (loc_mins - np.abs(x[idx])) ** 2 / 2 < np.abs(x[idx]) ** 2 / 2
        idx[idx] = global_min_idx
        out[idx] = np.sign(x[idx]) * loc_mins[global_min_idx]
        return out

    @staticmethod
    def _find_local_minima(b, c, d):
        f = -(c - b ** 2.0 / 3.0) ** 3.0 / 27.0
        g = (2.0 * b ** 3.0 - 9.0 * b * c + 27.0 * d) / 27.0
        idx = g ** 2.0 / 4.0 - f <= 0
        sqrtf = np.sqrt(f[idx])
        k = np.arccos(-(g[idx] / (2 * sqrtf)))
        loc_mins = 2 * sqrtf ** (1 / 3.0) * np.cos(k / 3.0) - b[idx] / 3.0
        return idx, loc_mins
