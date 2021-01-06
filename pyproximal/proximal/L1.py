import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import BoxProj
from pyproximal import ProxOperator


def _softthreshold(x, thresh, g=None):
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x - g``.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    thresh : :obj:`float`
        Threshold
    g : :obj:`numpy.ndarray`, optional
        Vector to subtract

    Returns
    -------
    x1 : :obj:`numpy.ndarray`
        Tresholded vector

    """
    if np.iscomplexobj(x):
        # https://stats.stackexchange.com/questions/357339/soft-thresholding-
        # for-the-lasso-with-complex-valued-data
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.exp(1j * np.angle(x))
    else:
        x1 = np.maximum(np.abs(x) - thresh, 0.) * np.sign(x)

    return x1


class L1(ProxOperator):
    r"""L1 norm proximal operator.

    Proximal operator of the L1 norm:
    :math:`\sigma||\mathbf{x} - \mathbf{g}||_1 = \sigma \sum |x_i - g_i|`.

    Parameters
    ----------
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L1 norm
    g : :obj:`np.ndarray`, optional
        Vector to be subtracted

    Notes
    -----
    The L1 proximal operator is defined as [1]_:

    .. math::

        prox_{\tau \sigma ||.||_1}(\mathbf{x}) =
        soft(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i - \tau \sigma, & x_i - g_i < -\tau \sigma \\
        g_i, & -\sigma \leq x_i - g_i \leq \tau\sigma \\
        x_i + \tau\sigma,  & x_i - g_i > \tau\sigma\\
        \end{cases}

    where ``soft`` is the so-called called *soft thresholding*.

    Moreover, as the conjugate of the L1 norm is the orthogonal projection of
    its dual norm (i.e., :math:`L_\inf` norm) onto a unit ball, its dual
    operator (when :math:`\mathbf{g}=\mathbf{0}`) is defined as:

    .. math::

        prox^*_{\tau \sigma ||.||_1}(\mathbf{x}) = P_{||.||_\inf <=\sigma} =
        \begin{cases}
        -\sigma, & x_i < -\sigma \\
        x_i,& -\sigma \leq x \leq \sigma \\
        \sigma,  & x_i > \sigma\\
        \end{cases}

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    def __init__(self, sigma=1., g=None):
        super().__init__(None, False)
        self.sigma = sigma
        self.g = g
        self.gdual = 0 if g is None else g
        self.box = BoxProj(-sigma, sigma)

    def __call__(self, x):
        return self.sigma * np.sum(np.abs(x))

    @_check_tau
    def prox(self, x, tau):
        if self.g is None:
            x = _softthreshold(x, tau * self.sigma)
        else:
            # use precomposition property
            x = _softthreshold(x - self.g, tau * self.sigma) + self.g
        return x

    @_check_tau
    def proxdual(self, x, tau):
        if not isinstance(self.gdual, np.ndarray):
            x = self.box(x)
        else:
            x = self._proxdual_moreau(x, tau)
        return x