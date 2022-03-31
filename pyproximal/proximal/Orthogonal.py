import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import MatrixMult, Identity
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Orthogonal(ProxOperator):
    r"""Proximal operator of any function of the product between an orthogonal
    matrix and a vector (plus summation of a vector).

    Proximal operator of any quadratic function
    :math:`g(\mathbf{x})=f(\mathbf{Qx} + \mathbf{b})` where :math:`\mathbf{Q}`
    is an orthogonal operator and :math:`\mathbf{b}` is a vector in the data
    space of :math:`\mathbf{Q}`.

    Parameters
    ----------
    f : :obj:`pyproximal.ProxOperator`
        Proximal operator
    Q : :obj:`pylops.LinearOperator`
        Orthogonal operator
    partial : :obj:`bool`, optional
        Partial (``True``) of full (``False``) orthogonality
    b : :obj:`np.ndarray`, optional
        Vector
    alpha : :obj:`float`, optional
        Positive coefficient for partial orthogonality. It will be ignored if
        ``partial=False``

    Notes
    -----
    The Proximal operator of any function of the form
    :math:`g(\mathbf{x}) = f(\mathbf{Qx} + \mathbf{b})` with the operator
    :math:`\mathbf{Q}` satisfying the following condition
    :math:`\mathbf{Q}\mathbf{Q}^T = \alpha \mathbf{I}` (partial orthogonality)
    is [1]_:

    .. math::

        \prox_{\tau g}(x) = \frac{1}{\alpha} ((\alpha \mathbf{I} -
        \mathbf{Q}^H \mathbf{Q}) \mathbf{x} + \mathbf{Q}^H
        (\prox_{\alpha \tau f}(\mathbf{Qx} + \mathbf{b}) - \mathbf{b}))

    A special case arises when :math:`\mathbf{Q}\mathbf{Q}^T =
    \mathbf{Q}^T\mathbf{Q} = \mathbf{I}`
    (full orthogonality), and the proximal operator reduces to:

    .. math::

        \prox_{\tau g}(x) = \mathbf{Q}^H (\prox_{\tau f}(\mathbf{Qx} +
        \mathbf{b}) - \mathbf{b}))

    .. [1] Daniel O'Connor, D., and Vandenberghe, L., "Primal-Dual
        Decomposition by Operator Splitting and Applications to Image
        Deblurring", SIAM J. Imaging Sciences, vol. 7, pp. 1724–1754. 2014.

    """
    def __init__(self, f, Q, partial=False, b=None, alpha=1.):
        super().__init__(None, False)
        self.f = f
        self.Q = Q
        self.partial = partial
        self.alpha = alpha
        self.b = b if b is not None else np.zeros(Q.shape[0], dtype=Q.dtype)

    def __call__(self, x):
        y = self.Q.matvec(x)
        if self.b is not None:
            y += self.b
        f = self.f(y)
        return f

    @_check_tau
    def prox(self, x, tau):
        y = self.Q.matvec(x)
        if self.partial:
            z = (1. / self.alpha) * \
                (self.alpha * x - self.Q.rmatvec(y) +
                 self.Q.rmatvec(self.f.prox(y + self.b, self.alpha * tau) -
                                self.b))
        else:
            if self.b is not None:
                y = y + self.b
            z = self.f.prox(y, tau)
            if self.b is not None:
                z = z - self.b
            z = self.Q.rmatvec(z)
        return z
