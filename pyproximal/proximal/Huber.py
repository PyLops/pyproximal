import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import BoxProj
from pyproximal import ProxOperator


class Huber(ProxOperator):
    r"""Huber norm proximal operator.

    Proximal operator of the Huber norm defined as:

    .. math::

        H(\mathbf{x})_\alpha =
        \begin{cases}
        \frac{||\mathbf{x}||_2^2}{2 \alpha}, & ||\mathbf{x}||_2 \leq \alpha \\
        ||\mathbf{x}||_1 - \alpha/2, & ||\mathbf{x}||_2 > \alpha \\
        \end{cases}

    which behaves like a L2 norm for :math:`|x_i| \leq \alpha` and a
    L1 norm for :math:`|x_i| < \alpha`.

    Parameters
    ----------
    alpha : :obj:`float`, optional
        Huber parameter

    Notes
    -----
    The Huber proximal operator is defined as:

    .. math::

        LOOK AT FOM SOLVER...

    """
    def __init__(self, alpha):
        super().__init__(None, False)
        self.alpha = alpha

    def __call__(self, x):
        l2 = np.linalg.norm(x)
        if l2 <= self:
            h =  l2 ** 2/ (2 * self.alpha)
        else:
            h =  np.sum(np.abs(x)) - self.alpha / 2.
        return h

    @_check_tau
    def prox(self, x, tau):
        x = (1. - tau / max(np.linalg.norm(x), tau + self.alpha)) * x
        return x

    @_check_tau
    def proxdual(self, x, tau):
        x = x / max(np.linalg.norm(x), self.alpha)
        return x