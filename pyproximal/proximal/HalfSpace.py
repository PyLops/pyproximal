import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import HalfSpaceProj


class HalfSpace(ProxOperator):
    r"""Euclidean ball proximal operator.

    Proximal operator of the half space: :math:`\operatorname{H}_{[w, b]} =
    \{ \mathbf{x}: \mathbf{w}^T \mathbf{x} \leq b \}`.

    Parameters
    ----------
    w : :obj:`np.ndarray`
        Coefficients of the half space
    b : :obj:`float`
        bias of the half space

    Notes
    -----
    As the half space is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.HalfSpaceProj` for details.

    """

    def __init__(self, w, b):
        super().__init__(None, False)
        self.w = w
        self.b = b
        self.half_space = HalfSpaceProj(self.w, self.b)

    def __call__(self, x):
        return np.dot(self.w, x) <= self.b

    @_check_tau
    def prox(self, x, tau):
        return self.half_space(x)
