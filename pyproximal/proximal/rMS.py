import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


def _l2(x, thresh):
    r"""Soft thresholding.

    Applies soft thresholding to vector ``x - g``.

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
    y = 1 / (1 + 2 * thresh) * x
    return y


def _current_sigma(sigma, count):
    if not callable(sigma):
        return sigma
    else:
        return sigma(count)


def _current_kappa(kappa, count):
    if not callable(kappa):
        return kappa
    else:
        return kappa(count)


class rMS(ProxOperator):
    r"""L1 norm proximal operator.

    Proximal operator of the :math:`\ell_1` norm:
    :math:`\sigma\|\mathbf{x} - \mathbf{g}\|_1 = \sigma \sum |x_i - g_i|`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L1 norm. This can be a constant number, a list
        of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``sigma`` to be used.
    g : :obj:`np.ndarray`, optional
        Vector to be subtracted

    Notes
    -----
    The :math:`\ell_1` proximal operator is defined as [1]_:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) =
        \operatorname{soft}(\mathbf{x}, \tau \sigma) =
        \begin{cases}
        x_i + \tau \sigma, & x_i - g_i < -\tau \sigma \\
        g_i, & -\tau\sigma \leq x_i - g_i \leq \tau\sigma \\
        x_i - \tau\sigma,  & x_i - g_i > \tau\sigma\\
        \end{cases}

    where :math:`\operatorname{soft}` is the so-called called *soft thresholding*.

    Moreover, as the conjugate of the :math:`\ell_1` norm is the orthogonal projection of
    its dual norm (i.e., :math:`\ell_\inf` norm) onto a unit ball, its dual
    operator (when :math:`\mathbf{g}=\mathbf{0}`) is defined as:

    .. math::

        \prox^*_{\tau \sigma \|\cdot\|_1}(\mathbf{x}) = P_{\|\cdot\|_{\infty} <=\sigma}(\mathbf{x}) =
        \begin{cases}
        -\sigma, & x_i < -\sigma \\
        x_i,& -\sigma \leq x_i \leq \sigma \\
        \sigma,  & x_i > \sigma\\
        \end{cases}

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    def __init__(self, sigma=1., kappa=1., g=None):
        super().__init__(None, False)
        self.sigma = sigma
        self.kappa = kappa
        self.g = g
        self.gdual = 0 if g is None else g
        self.count = 0

    def __call__(self, x):
        sigma = _current_sigma(self.sigma, self.count)
        kappa = _current_sigma(self.kappa, self.count)
        return np.minimum(sigma * np.linalg.norm(x)**2, kappa)

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
        kappa = _current_sigma(self.kappa, self.count)

        x = np.where(np.abs(x) <= np.sqrt(kappa / sigma * (1 + 2 * tau * sigma)), _l2(x, tau * sigma), x)
        return x

    @_check_tau
    def proxdual(self, x, tau):
        x - tau * self.prox(x / tau, 1. / tau)
        # x = self._proxdual_moreau(x, tau)

        return x