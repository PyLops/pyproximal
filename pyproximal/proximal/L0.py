import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import L0BallProj
from pyproximal import ProxOperator


class L0Ball(ProxOperator):
    r"""L0 ball proximal operator.

    Proximal operator of the L0 ball: :math:`L0_{r} =
    \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}`.

    Parameters
    ----------
    radius : :obj:`float`
        Radius

    Notes
    -----
    As the L0 ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.L0BallProj` for details.

    """
    def __init__(self, radius):
        super().__init__(None, False)
        self.radius = radius
        self.ball = L0BallProj(self.radius)

    def __call__(self, x, tol=1e-4):
        return np.linalg.norm(np.abs(x), ord=0) <= self.radius

    @_check_tau
    def prox(self, x, tau):
        return self.ball(x)
