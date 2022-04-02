import numpy as np

from scipy.sparse.linalg import lsqr
from pylops import MatrixMult, Identity
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Quadratic(ProxOperator):
    r"""Quadratic function proximal operator.

    Proximal operator for a quadratic function: :math:`f(\mathbf{x}) =
    \frac{1}{2} \mathbf{x}^T \mathbf{Op} \mathbf{x} + \mathbf{b}^T
    \mathbf{x} + c`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator (must be square)
    b : :obj:`np.ndarray`, optional
        Vector
    c : :obj:`float`, optional
        Scalar
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme used to compute the proximal
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.

    Raises
    ------
    ValueError
        If ``Op`` is not square

    Notes
    -----
    The Quadratic proximal operator is defined as:

    .. math::

        \prox_{\tau f}(\mathbf{x}) =
        \left(\mathbf{I} + \tau  \mathbf{Op} \right)^{-1} \left(\mathbf{x} -
        \tau \mathbf{b}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, the proximal operator reduces to:

    .. math::

        \prox_{\mathbf{b}^T \mathbf{x} + c}(\mathbf{x}) =
        \mathbf{x} - \tau \mathbf{b}

    Finally if also ``b`` is not provided, the proximal operator of a constant
    function simply becomes :math:`\prox_c(\mathbf{x}) = \mathbf{x}`


    """
    def __init__(self, Op=None, b=None, c=0., niter=10, x0=None, warm=True):
        if Op is not None:
            if Op.shape[0] != Op.shape[1]:
                raise ValueError('Op must be square')
        super().__init__(Op, True)
        self.b = b
        if self.Op is not None and self.b is None:
            self.b = np.zeros(Op.shape[1], dtype=Op.dtype)
        self.c = c
        self.niter = niter
        self.x0 = x0
        self.warm = warm

    def __call__(self, x):
        if self.Op is not None and self.b is not None:
            f = np.dot(x, self.Op * x) / 2. + np.dot(self.b, x) + self.c
        elif self.b is not None:
            f = np.dot(self.b, x) + self.c
        else:
            f = self.c
        return f

    @_check_tau
    def prox(self, x, tau):
        if self.Op is not None and self.b is not None:
            y = x - tau * self.b
            if self.Op.explicit:
                Op1 = MatrixMult(np.eye(self.Op.shape[0]) + tau * self.Op.A)
                x = Op1.div(y)
            else:
                Op1 = Identity(self.Op.shape[0], dtype=self.Op.dtype) + \
                      tau * self.Op.A
                x = lsqr(Op1, y, iter_lim=self.niter, x0=self.x0)[0]
            if self.warm:
                self.x0 = x
        elif self.b is not None:
            x = x - tau * self.b
        return x

    def grad(self, x):
        """Compute gradient

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector

        Returns
        -------
        g : :obj:`np.ndarray`
            Gradient vector

        """
        if self.Op is not None and self.b is not None:
            g = self.Op.matvec(x) / 2. + x
        elif self.b is not None:
            g = x
        else:
            g = 0.
        return g