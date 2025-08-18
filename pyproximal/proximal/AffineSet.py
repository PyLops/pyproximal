from typing import TYPE_CHECKING

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal import ProxOperator
from pyproximal.projection import AffineSetProj
from pyproximal.ProxOperator import _check_tau

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class AffineSet(ProxOperator):
    r"""Affine set proximal operator.

    Proximal operator of an Affine set: :math:`\{ \mathbf{x} : \mathbf{Opx}=\mathbf{b} \}`.

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
    As the Affine set is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.AffineSetProj` for
    details.

    """

    def __init__(self, Op: "LinearOperator", b: NDArray, niter: int) -> None:
        super().__init__(Op, False)
        self.b = b
        self.niter = niter
        self.affine = AffineSetProj(self.Op, self.b, self.niter)

    def __call__(self, x: NDArray) -> bool:
        if np.allclose(self.Op.matvec(x), self.b):
            return True
        else:
            return False

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        return self.affine(x)
