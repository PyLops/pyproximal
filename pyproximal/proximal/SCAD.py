import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class SCAD(ProxOperator):
    r"""Smoothly clipped absolute deviation (SCAD) penalty.

    The SCAD penalty is a concave function and is defined as

    .. math::

        \mathrm{SCAD}_{\sigma, a}(\mathbf{x}) =
        \begin{cases}
        \sigma x_i, & |x_i| \leq \sigma \\
        \frac{-x_i^2 + 2 a \sigma  - \sigma^2}{2 (a - 1)}, & \sigma < |x_i| \leq a\sigma \\
        \frac{(a + 1)\sigma^2}{2}, & |x_i| > a\sigma
        \end{cases}

    Parameters
    ----------
    sigma : :obj:`float`
        First threshold parameter (named :math:`\lambda` in the original paper [1]_)
    a : :obj:`float`, optional
        Second threshold parameter (must be larger than 2). Default is 3.7, see [1]_ for more information.

    Notes
    -----

    The SCAD penalty is continuous and differentiable and does not suffer from
    biasedness as the :math:`\ell_1`-norm, nor is discontinuous as the hard
    thresholding penalty. Thus, it fuses the most favourable properties of both
    penalties.

    The proximal operator is given by

    .. math::

        \prox_{\tau \mathrm{SCAD}_{\sigma, a}(\cdot)}(\mathbf{x}) =
        \begin{cases}
        \sgn(x_i)\max(0, |x_i| - \sigma), & |x_i| \leq \frac{\sigma(a - 1 - \tau + a\tau)}{a - 1} \\
        \frac{(a-1)x_i - \sgn(x_i)a\tau\sigma}{a-1-\tau}, & \frac{\sigma(a - 1 - \tau + a\tau)}{a - 1} < |x_i| \leq a\sigma \\
        x_i, & |x_i| > a\sigma
        \end{cases}

    .. [1] Fan, J. and Li, R. "Variable selection via nonconcave penalized likelihood and its oracle
        properties" Journal of the American Statistical Association, 96(456):1348–1360, 2001

    """

    def __init__(self, sigma, a=3.7):
        super().__init__(None, False)
        self.sigma = sigma
        if sigma <= 0:
            raise ValueError('Variable "sigma" must be positive.')
        if a <= 2:
            raise ValueError('Variable "a" must be larger than two.')
        self.a = a

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        f = np.zeros_like(x)
        absx = np.abs(x)
        ind = absx <= self.sigma
        f[ind] = self.sigma * absx[ind]
        ind = np.logical_and(self.sigma < absx, absx <= self.a * self.sigma)
        f[ind] = (-x[ind] ** 2 + 2 * self.a * self.sigma * absx[ind] - self.sigma ** 2) / (2 * (self.a - 1))
        ind = absx > self.a * self.sigma
        f[ind] = (self.a + 1) * self.sigma ** 2 / 2
        return f

    @_check_tau
    def prox(self, x, tau):
        theta = x.copy()
        absx = np.abs(x)
        first_threshold = self.sigma * (self.a - 1 - tau + tau * self.a) / (self.a - 1)
        ind = absx <= first_threshold
        theta[ind] = np.sign(x[ind]) * np.maximum(0, absx[ind] - self.sigma * tau)
        ind = np.logical_and(first_threshold < absx, absx <= self.a * self.sigma)
        theta[ind] = ((self.a - 1) * x[ind] - np.sign(x[ind]) * self.a * self.sigma * tau) / (self.a - 1 - tau)
        return theta
