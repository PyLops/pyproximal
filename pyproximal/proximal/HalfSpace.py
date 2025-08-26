import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.projection.HalfSpace import HalfSpaceProj
from pyproximal.ProxOperator import ProxOperator, _check_tau


class HalfSpace(ProxOperator):
    r"""Euclidean ball proximal operator.

    Proximal operator of the half space: :math:`\operatorname{H}_{[w, b]} =
    \{ \mathbf{x}: \mathbf{w}^T \mathbf{x} \leq b \}`.

    Parameters
    ----------
    w : :obj:`numpy.ndarray`
        Coefficients of the half space
    b : :obj:`float`
        bias of the half space

    Notes
    -----
    As the half space is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.HalfSpaceProj` for details.

    """

    def __init__(self, w: NDArray, b: float) -> None:
        super().__init__(None, False)
        self.w = w
        self.b = b
        self.half_space = HalfSpaceProj(self.w, self.b)

    def __call__(self, x: NDArray) -> bool:
        return bool(np.dot(self.w, x) <= self.b)

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        return self.half_space(x)
