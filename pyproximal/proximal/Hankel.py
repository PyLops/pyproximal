import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import HankelProj


class Hankel(ProxOperator):
    r"""Hankel proximal operator.

    Proximal operator of the Hankel matrix indicator function.

    Parameters
    ----------
    dim : :obj:`tuple`
        Dimension of the Hankel matrix.

    Notes
    -----
    As the Hankel Operator is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.HankelProj` for
    details).

    """
    def __init__(self, dim):
        super().__init__(None, False)
        self.dim = dim
        self.hankel_proj = HankelProj()

    def __call__(self, x):
        X = x.reshape(self.dim)
        return np.allclose(X, self.hankel_proj(X))

    @_check_tau
    def prox(self, x, tau):
        X = x.reshape(self.dim)
        return self.hankel_proj(X).ravel()
