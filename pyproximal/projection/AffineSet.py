import numpy as np
from scipy.sparse.linalg import lsqr


class AffineSetProj():
    r"""Affine set projection.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator
    b : :obj:`numpy.ndarray`
        Data vector
    niter : :obj:`int`
        Number of iterations of iterative scheme used to compute the projection.

    Notes
    -----
    Given an Affine set defined as:

    .. math::

        \{ \mathbf{x} : \mathbf{Opx}=\mathbf{b} \}

    its orthogonal projection is:

    .. math::

       P_{\{\mathbf{y}:\mathbf{Opy}=\mathbf{b}\}} (\mathbf{x}) = \mathbf{x} -
       \mathbf{Op}^H(\mathbf{Op}\mathbf{Op}^H)^{-1}(\mathbf{Opx}-\mathbf{b})

    Note the this is the proximal operator of the corresponding
    indicator function :math:`I_{\{\mathbf{Opx}=\mathbf{b}\}}`

    """
    def __init__(self, Op, b, niter):
        self.Op = Op
        self.b = b
        self.niter = niter

    def __call__(self, x):
        inv = lsqr(self.Op * self.Op.H, self.Op * x - self.b, iter_lim=self.niter)[0]
        y = x - self.Op.H * inv
        return y