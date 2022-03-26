import numpy as np
from scipy.special import lambertw

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class ETP(ProxOperator):
    r"""Exponential-type penalty (ETP).

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.0.

    Notes
    -----
    The exponential-type penalty is defined as

    .. math::

        ETP_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma}{1-e^{-\gamma}}(1-e^{-\gamma|x_i|})

    for :math:`{\sigma>0}`, and :math:`{\gamma\geq 0}`. Note that when
    :math:`{\gamma\rightarrow 0}` the logarithmic penalty approaches the l1-penalty and when
    :math:`{\gamma\rightarrow\infty}` tends to the l0-penalty [1]_.

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
        for i, y in enumerate(x):
            tmp = np.exp(-np.abs(y) * self.gamma) * k * self.gamma ** 2
            if tmp <= np.exp(-1):
                stat_point = np.sign(y) * np.real(lambertw(-tmp)) / self.gamma + y
                if tau * self.elementwise(stat_point) + (stat_point - y) ** 2 / 2 < y ** 2 / 2:
                    out[i] = stat_point
        return out
