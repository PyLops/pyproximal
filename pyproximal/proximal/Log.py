import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Log(ProxOperator):
    r"""Logarithmic penalty.

    Parameters
    ----------
    sigma : :obj:`float`
        Regularization parameter.
    gamma : :obj:`float`, optional
        Regularization parameter. Default is 1.3.

    Notes
    -----
    The logarithmic penalty is an extension of the elastic net family of penalties to non-convex members, which
    should produce sparser solutions compared to the l1-penalty [1]_. The pyproximal implementation considers a scaled
    version where

    .. math::

        Log_{\sigma,\gamma}(\mathbf{x}) = \sum_i \frac{\sigma}{\log(\gamma + 1)}\log(\gamma|x_i| + 1)

    where :math:`{\sigma>0}`, :math:`{\gamma>0}`. This satisfies :math:`{Log_{\sigma,\gamma}(0) = 0}` and
    :math:`{Log_{\sigma,\gamma}(1) = \sigma}`, which is suitable also for penalizing singular values. Note that when
    :math:`{\gamma\rightarrow 0}` the logarithmic penalty approaches the l1-penalty and when
    :math:`{\gamma\rightarrow\infty}` it mimicks the l0-penalty.

    .. [1] Friedman, J. H. "Fast sparse regression and classification",
        International Journal of Forecasting, 28(3):722 – 738, 2012.

    """

    def __init__(self, sigma, gamma=1.3):
        super().__init__(None, False)
        if sigma < 0:
            raise ValueError('Variable "sigma" must be positive.')
        if gamma < 0:
            raise ValueError('Variable "gamma" must be positive.')
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.sigma / np.log(self.gamma + 1) * np.log(self.gamma * np.abs(x) + 1)

    @_check_tau
    def prox(self, x, tau):
        k = tau * self.sigma / np.log(self.gamma + 1)
        out = np.zeros_like(x)
        for i, y in enumerate(x):
            b = self.gamma * np.abs(y) - 1
            discriminant = b ** 2 - 4 * self.gamma * (k * self.gamma - np.abs(y))
            if discriminant >= 0:
                c = np.sqrt(discriminant)
                r = np.array([0, (b - c) / (2 * self.gamma), (b + c) / (2 * self.gamma)])
                val = tau * self.elementwise(r) + (r - np.abs(y)) ** 2 / 2
                idx = np.argmin(val)
                out[i] = r[idx]
                out[i] *= np.sign(y)
        return out