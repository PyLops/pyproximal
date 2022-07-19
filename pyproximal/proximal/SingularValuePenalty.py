import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class SingularValuePenalty(ProxOperator):
    r"""Proximal operator of a penalty acting on the singular values.

    Generic regularizer :math:`\mathcal{R}_f` acting on the singular values of a matrix,

    .. math::

        \mathcal{R}_f(\mathbf{X}) = f(\boldsymbol\lambda)

    where :math:`\mathbf{X}` is a matrix of size :math:`M \times N` and
    :math:`\boldsymbol\lambda` is the corresponding singular value vector.

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`.
    penalty : :obj:`pyproximal.ProxOperator`
        Function acting on the singular values.

    Notes
    -----
    The pyproximal implementation allows ``penalty`` to be any
    :class:`pyproximal.ProxOperator` acting on the singular values; however, not all
    penalties will result in a mathematically accurate proximal operator defined this
    way. Given a penalty :math:`f`, the proximal operator is assumed to be

    .. math::

        \prox_{\tau \mathcal{R}_f}(\mathbf{X}) =
        \mathbf{U} \diag\left( \prox_{\tau f}(\boldsymbol\lambda)\right) \mathbf{V}^H

    where :math:`\mathbf{X} = \mathbf{U}\diag(\boldsymbol\lambda)\mathbf{V}^H`, is an
    SVD of :math:`\mathbf{X}`. It is the user's responsibility to check that this is
    true for their particular choice of ``penalty``.
    """

    def __init__(self, dim, penalty):
        super().__init__(None, False)
        self.dim = dim
        self.penalty = penalty

    def __call__(self, x):
        X = x.reshape(self.dim)
        eigs = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0  # ensure all eigenvalues at positive
        return np.sum(self.penalty(np.sqrt(eigs)))

    @_check_tau
    def prox(self, x, tau):
        X = x.reshape(self.dim)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        X = np.dot(U * self.penalty.prox(S, tau), Vh)
        return X.ravel()
