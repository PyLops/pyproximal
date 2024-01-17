from scipy.sparse.linalg import lsqr as sp_lsqr
from scipy.sparse.linalg import cg as sp_cg
from pylops.optimization.basic import cg, lsqr
from pylops.utils.backend import get_array_module, get_module_name


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
        if get_module_name(get_array_module(x)) == 'numpy':
            inv = sp_cg(self.Op * self.Op.H, self.Op * x - self.b, maxiter=self.niter)[0]
        else:
            inv = cg(self.Op * self.Op.H, self.Op * x - self.b, niter=self.niter)[0]
        y = x - self.Op.H * inv.ravel() # currently ravel is added to ensure that the output is always a vector
        return y