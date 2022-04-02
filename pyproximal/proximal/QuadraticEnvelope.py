import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class QuadraticEnvelopeCard(ProxOperator):
    r"""Quadratic envelope of the :math:`\ell_0`-penalty.

    The :math:`\ell_0`-penalty is also known as the *cardinality function*, and the
    quadratic envelope :math:`\mathcal{Q}(\mu\|\cdot\|_0)` of it is defined as

    .. math::

        \mathcal{Q}(\mu\|\cdot\|_0)(x) = \sum_i \left(\mu - \frac{1}{2}\max(0, \sqrt{2\mu} - |x_i|)^2\right)

    where :math:`\mu \geq 0`.

    Parameters
    ----------
    mu : :obj:`float`
        Threshold parameter.

    Notes
    -----
    The terminology *quadratic envelope* was coined in [1]_, however, the rationale has
    been used earlier, e.g. in [2]_. In a general setting, the quadratic envelope
    :math:`\mathcal{Q}(f)(x)` is defined such that

    .. math::

        \left(f(x) + \frac{1}{2}\|x-y\|_2^2\right)^{**} = \mathcal{Q}(f)(x) + \frac{1}{2}\|x-y\|_2^2

    where :math:`g^{**}` denotes the bi-conjugate of :math:`g`, which is the l.s.c.
    convex envelope of :math:`g`.

    There is no closed-form expression for :math:`\mathcal{Q}(f)(x)` given an arbitrary
    function :math:`f`. However, for certain special cases, such as in the case of the
    cardinality function, such expressions do exist.

    The proximal operator is given by

    .. math::

        \prox_{\tau\mathcal{Q}(\mu\|\cdot\|_0)}(x) =
        \begin{cases}
        x_i, & |x_i| \geq \sqrt{2 \mu} \\
        \frac{x_i-\tau\sqrt{2\mu}\sgn(x_i)}{1-\tau}, & \tau\sqrt{2\mu} < |x_i| < \sqrt{2 \mu} \\
        0, & |x_i| \leq \tau\sqrt{2 \mu}
        \end{cases}

    By inspecting the structure of the proximal operator it is clear that large values
    are unaffected, whereas smaller ones are penalized partially or completely. Such
    properties are desirable to counter the effect of *shrinking bias* observed with
    e.g. the :math:`\ell_1`-penalty. Note that in the limit :math:`\tau=1` this becomes
    the hard thresholding with threshold :math:`\sqrt{2\mu}`. It should also be noted
    that this proximal operator is identical to the Minimax Concave Penalty (MCP)
    proposed in [3]_.

    .. [1] Carlsson, M. "On Convex Envelopes and Regularization of Non-convex
        Functionals Without Moving Global Minima", In Journal of Optimization Theory
        and Applications, 183:66–84, 2019.
    .. [2] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.
    .. [3] Zhang et al. "Nearly unbiased variable selection under minimax concave
        penalty", In the Annals of Statistics, 38(2):894–942, 2010.

    """

    def __init__(self, mu):
        super().__init__(None, False)
        self.mu = mu

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.mu - 0.5 * np.maximum(0, np.sqrt(2 * self.mu) - np.abs(x)) ** 2

    @_check_tau
    def prox(self, x, tau):
        r = np.abs(x)
        idx = r < np.sqrt(2 * self.mu)
        if tau >= 1:
            r[idx] = 0
        else:
            r[idx] = np.maximum(0, (r[idx] - tau * np.sqrt(2 * self.mu)) / (1 - tau))
        return r * np.sign(x)
