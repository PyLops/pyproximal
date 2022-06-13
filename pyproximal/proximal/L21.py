import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class L21(ProxOperator):
    r""":math:`L_{2,1}` proximal operator.

    Proximal operator for :math:`L_{2,1}` matrix norm.

    Parameters
    ----------
    ndim : :obj:`int`
        Number of dimensions :math:`N_{dim}`. Used to reshape the input array
        in a matrix of size :math:`N_{dim} \times N'_{x}` where
        :math:`N'_x = \frac{N_x}{N_{dim}}`. Note that the input
        vector ``x`` should be created by stacking vectors from different
        dimensions.
    sigma : :obj:`float`, optional
        Multiplicative coefficient of :math:`L_{2,1}` norm

    Notes
    -----
    Given the :math:`L_{2,1}` norm of a matrix of size
    :math:`N_{dim} \times N'_x` defined as:

    .. math::

        \sigma \|\mathbf{X}\|_{2,1} = \sigma \sum_{j=0}^{N'_x} \|\mathbf{x}_j\|_2 =
        \sigma \sum_{j=0}^{N'_x} \sqrt{\sum_{i=0}^{N_{dim}}} |x_{ij}|^2

    the proximal operator is:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_{2,1}}(\mathbf{x}_j) =
        \left(1 - \frac{\sigma \tau}{max\{||\mathbf{x}_j||_2,
        \sigma \tau \}}\right) \mathbf{x}_j \quad \forall j

    Similar to the Euclidean norm, the dual operator is defined as:

    .. math::

        \prox^*_{\tau \sigma \||\cdot\|_{2,1}}(\mathbf{x}_j) =
        \frac{\sigma \mathbf{x}_j}{\max\{||\mathbf{x}_j||_2, \sigma\}}
        \quad \forall j

    Finally, we note that the :math:`L_{2,1}` norm is a separable function
    on each column on the matrix :math:`\mathbf{X}`. Taking advantage of the
    property of proximal operator of separable function [1]_, its proximal and
    dual proximal operators can be interpreted as a series of
    :class:`pyproximal.proximal.Euclidean` operators on each column
    of the matrix :math:`\mathbf{X}`.

    .. [1] N., Parikh, "Proximal Algorithms", Foundations and Trends
        in Optimization. 2013.

    """
    def __init__(self, ndim, sigma=1.):
        super().__init__(None, False)
        self.ndim = ndim
        self.sigma = sigma

    def __call__(self, x):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        f = self.sigma * np.sum(np.sqrt(np.sum(x ** 2, axis=0)))
        return f

    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        aux = np.sqrt(np.sum(x ** 2, axis=0))
        aux = np.vstack([aux] * self.ndim).ravel()
        x = (1 - (tau * self.sigma) / np.maximum(aux, tau * self.sigma)) * x.ravel()
        return x

    @_check_tau
    def proxdual(self, x, tau):
        x = x.reshape(self.ndim, len(x) // self.ndim)
        aux = np.sqrt(np.sum(x ** 2, axis=0))
        aux = np.vstack([aux] * self.ndim).ravel()
        x = self.sigma * x.ravel() / np.maximum(aux, self.sigma)
        return x