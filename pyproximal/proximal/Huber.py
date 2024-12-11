import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.proximal import L2, L1


class Huber(ProxOperator):
    r"""Huber norm proximal operator.

    Proximal operator of the Huber norm defined as 
    :math:`H_\alpha(\mathbf{x}) = \sum_i H_\alpha(x_i)` where:

    .. math::

        H_\alpha(x_i) = 
        \begin{cases}
        \frac{|x_i|^2}{2 \alpha}, & |x_i| \leq \alpha \\
        |x_i| - \frac{\alpha}{2}, & |x_i| > \alpha
        \end{cases}

    which behaves like a :math:`\ell_2` norm for :math:`|x_i| \leq \alpha` and a
    :math:`\ell_1` norm for :math:`|x_i| > \alpha`.

    Parameters
    ----------
    alpha : :obj:`float`
        Huber parameter

    Notes
    -----
    The Huber proximal operator is defined as:

    .. math::

        \prox_{\tau H_\alpha(\cdot)}(\mathbf{x}) =
        \begin{cases}
        \prox_{\frac{\tau}{2 \alpha} |x_i|^2}(x_i), & |x_i| \leq \alpha \\
        \prox_{\tau |x_i|}(x_i), & |x_i| > \alpha
        \end{cases}
        
    """
    def __init__(self, alpha):
        super().__init__(None, False)
        self.alpha = alpha
        self.l2 = L2(sigma=1. / self.alpha)
        self.l1 = L1()

    def __call__(self, x):
        h = np.zeros_like(x)
        xabs = np.abs(x)
        mask = xabs > self.alpha
        h[~mask] = xabs[~mask] ** 2 / (2. * self.alpha)
        h[mask] = xabs[mask] - self.alpha / 2.
        return np.sum(h)

    @_check_tau
    def prox(self, x, tau):
        y = np.zeros_like(x)
        xabs = np.abs(x)
        mask = xabs > self.alpha
        y[~mask] = self.l2.prox(x[~mask], tau)
        y[mask] = self.l1.prox(x[mask], tau)
        # alternative from https://math.stackexchange.com/questions/1650411/
        # proximal-operator-of-the-huber-loss-function... currently commented
        # as it does not provide the same result
        # y = (1. - tau / np.maximum(np.abs(x), tau + self.alpha)) * x

        return y
    

class HuberCircular(ProxOperator):
    r"""Circular Huber norm proximal operator.

    Proximal operator of the Circular Huber norm defined as:

    .. math::

        H_\alpha(\mathbf{x}) =
        \begin{cases}
        \frac{\|\mathbf{x}\|_2^2}{2 \alpha}, & \|\mathbf{x}\|_2 \leq \alpha \\
        \|\mathbf{x}\|_2 - \frac{\alpha}{2}, & \|\mathbf{x}\|_2 > \alpha \\
        \end{cases}

    which behaves like a :math:`\ell_2` norm for :math:`\|\mathbf{x}\|_2 \leq \alpha` and a
    :math:`\ell_1` norm for :math:`\|\mathbf{x}\|_2 > \alpha`.

    Parameters
    ----------
    alpha : :obj:`float`
        Huber parameter

    Notes
    -----
    The Circular Huber proximal operator is defined as [1]_:

    .. math::

        \prox_{\tau H_\alpha(\cdot)}(\mathbf{x}) =
        \left( 1 - \frac{\tau}{\max\{\|\mathbf{x}\|_2, \tau + \alpha \} } \right) \mathbf{x}

    .. [1] O’Donoghue, B. and Stathopoulos, G. and Boyd, S. "A Splitting Method for Optimal Control", 
        In the IEEE Transactions on Control Systems Technology, 2013.
        
    """
    def __init__(self, alpha):
        super().__init__(None, False)
        self.alpha = alpha

    def __call__(self, x):
        l2 = np.linalg.norm(x)
        if l2 <= self.alpha:
            h = l2 ** 2 / (2 * self.alpha)
        else:
            h = l2 - self.alpha / 2.
        return h

    @_check_tau
    def prox(self, x, tau):
        x = (1. - tau / max(np.linalg.norm(x), tau + self.alpha)) * x
        return x
