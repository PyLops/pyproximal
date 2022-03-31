import numpy as np
from scipy.special import lambertw

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class ETP(ProxOperator):
    r"""Exponential-type penalty (ETP).

    The exponential-type penalty is defined as

    .. math::

        \mathrm{ETP}_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma}{1-e^{-\gamma}}(1-e^{-\gamma|x_i|})

    for :math:`{\sigma>0}`, and :math:`{\gamma>0}`.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.0.

    Notes
    -----
    As :math:`{\gamma\rightarrow 0}` the exponential-type penalty approaches the :math:`\ell_1`-penalty and when
    :math:`{\gamma\rightarrow\infty}` tends to the :math:`\ell_0`-penalty [1]_.

    As for the proximal operator, consider the one-dimensional case

    .. math::

        \prox_{\tau \mathrm{ETP}(\cdot)}(x) = \argmin_{z} \mathrm{ETP}(z) + \frac{1}{2\tau}(x - z)^2

    and assume that :math:`x\geq 0`. The minima can be obtained when :math:`z=0` or at a stationary point,
    where the latter must satisfy

    .. math::

        x = z + \frac{\gamma \sigma \tau}{1-e^{-\gamma}} e^{-\gamma z} .

    The solution to the above equation can be expressed using the *Lambert W function*.

    .. [1] Gao, C. et al. "A Feasible Nonconvex Relaxation Approach to Feature Selection",
        In the Proceedings of the Conference on Artificial Intelligence (AAAI), 2011.

    """

    def __init__(self, sigma, gamma=1.0):
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
        return self.sigma / (1 - np.exp(-self.gamma)) * (1 - np.exp(-self.gamma * np.abs(x)))

    @_check_tau
    def prox(self, x, tau):
        k = tau * self.sigma / (1 - np.exp(-self.gamma))
        out = np.zeros_like(x)

        # Get real-valued solutions to the Lambert W function
        tmp = np.exp(-np.abs(x) * self.gamma) * k * self.gamma ** 2
        idx = tmp <= np.exp(-1)
        stat_points = np.sign(x[idx]) * np.real(lambertw(-tmp[idx])) / self.gamma + x[idx]

        # Check which stationary points are global minima
        idx_minima = tau * self.elementwise(stat_points) + (stat_points - x[idx]) ** 2 / 2 < x[idx] ** 2 / 2
        idx[idx] = idx_minima
        out[idx] = stat_points[idx_minima]
        return out
