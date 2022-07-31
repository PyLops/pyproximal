import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import L0BallProj
from pyproximal import ProxOperator
from pyproximal.proximal.L1 import _current_sigma


def _hardthreshold(x, thresh):
    r"""Hard thresholding.

    Applies hard thresholding to vector ``x`` (equal to the proximity
    operator for :math:`\|\mathbf{x}\|_0`) as shown in [1]_.

    .. [1] Chen, F., Shen, L., Suter, B.W., "Computing the proximity
       operator of the Lp norm with 0 < p < 1",
       IET Signal Processing, 10, 2016.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    x1 = x.copy()
    x1[np.abs(x) <= thresh] = 0
    return x1


class L0(ProxOperator):
    r"""L0 norm proximal operator.

    Proximal operator of the :math:`\ell_0` norm:
    :math:`\sigma\|\mathbf{x}\|_0 = \text{count}(x_i \ne 0)`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L0 norm. This can be a constant number, a list
        of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``sigma`` to be used.

    Notes
    -----
    The :math:`\ell_0` proximal operator is defined as:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_0}(\mathbf{x}) =
        \operatorname{hard}(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i, & x_i < -\tau \sigma \\
        0, & -\tau\sigma \leq x_i \leq \tau\sigma \\
        x_i,  & x_i > \tau\sigma\\
        \end{cases}

    where :math:`\operatorname{hard}` is the so-called called *hard thresholding*.

    """
    def __init__(self, sigma=1.):
        super().__init__(None, False)
        self.sigma = sigma
        self.count = 0

    def __call__(self, x):
        sigma = _current_sigma(self.sigma, self.count)
        return np.sum(np.abs(x) > sigma)

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        sigma = _current_sigma(self.sigma, self.count)
        x = _hardthreshold(x, tau * sigma)
        return x


class L0Ball(ProxOperator):
    r"""L0 ball proximal operator.

    Proximal operator of the L0 ball: :math:`L0_{r} =
    \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}`.

    Parameters
    ----------
    radius : :obj:`int` or :obj:`func`, optional
        Radius. This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has been
        invoked before and returns a scalar ``radius`` to be used.
        Radius

    Notes
    -----
    As the L0 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L0BallProj` for details.

    """
    def __init__(self, radius):
        super().__init__(None, False)
        self.radius = radius
        self.ball = L0BallProj(self.radius if not callable(radius) else radius(0))
        self.count = 0

    def __call__(self, x, tol=1e-4):
        radius = _current_sigma(self.radius, self.count)
        return np.linalg.norm(np.abs(x), ord=0) <= radius

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        radius = _current_sigma(self.radius, self.count)
        self.ball.radius = radius
        y = self.ball(x)
        return y