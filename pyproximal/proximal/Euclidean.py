import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import EuclideanBallProj


class Euclidean(ProxOperator):
    r"""Euclidean norm proximal operator.

    Proximal operator of the Euclidean norm: :math:`\sigma \|\mathbf{x}\|_2 =
    \sigma \sqrt{\sum x_i^2}`.

    Parameters
    ----------
    sigma : :obj:`int`, optional
        Multiplicative coefficient of :math:`L_{2}` norm

    Notes
    -----
    The Euclidean proximal operator is defined as:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_2}(\mathbf{x}) =
        \left(1 - \frac{\tau \sigma }{\max\{\|\mathbf{x}\|_2,
        \tau \sigma \}}\right) \mathbf{x}

    This operator is sometimes called *block soft thresholding*.

    Moreover, as the conjugate of the Euclidean norm is the orthogonal
    projection of its dual norm (i.e., Euclidean norm) onto a unit ball,
    its dual operator is defined as:

    .. math::

        \prox^*_{\tau \sigma \|\cdot\|_2}(\mathbf{x}) =
        \frac{\sigma \mathbf{x}}{\max\{\|\mathbf{x}\|_2, \sigma\}}

    """
    def __init__(self, sigma=1.):
        super().__init__(None, True)
        self.sigma = sigma

    def __call__(self, x):
        return self.sigma * np.linalg.norm(x)

    @_check_tau
    def prox(self, x, tau):
        x = (1. - (tau * self.sigma) / max(np.linalg.norm(x), tau * self.sigma)) * x
        return x

    @_check_tau
    def proxdual(self, x, tau):
        x = self.sigma * x / (max(np.linalg.norm(x), self.sigma))
        return x

    def grad(self, x):
        return self.sigma * x / np.linalg.norm(x)


class EuclideanBall(ProxOperator):
    r"""Euclidean ball proximal operator.

    Proximal operator of the Euclidean ball: :math:`Eucl_{[c, r]} =
    \{ \mathbf{x}: ||\mathbf{x} - \mathbf{c}||_2 \leq r \}`.

    Parameters
    ----------
    center : :obj:`np.ndarray` or :obj:`float`
        Center of the ball
    radius : :obj:`float`
        Radius

    Notes
    -----
    As the Euclidean ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.EuclideanBallProj` for details.

    """
    def __init__(self, center, radius):
        super().__init__(None, False)
        self.center = center
        self.radius = radius
        self.ball = EuclideanBallProj(self.center, self.radius)

    def __call__(self, x):
        return np.linalg.norm(x - self.center) <= self.radius

    @_check_tau
    def prox(self, x, tau):
        return self.ball(x)
