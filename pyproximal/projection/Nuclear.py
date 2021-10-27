import numpy as np
from pyproximal.projection import L1BallProj


class NuclearBallProj():
    r"""Nuclear ball projection

    Parameters
    ----------
    n : :obj:`int`
        Number of elements of input vector
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    Given a Nuclear ball defined as:

    .. math::

        N_{r} = \{ \mathbf{X}: ||\mathbf{X}||_* \leq r \}

    its orthogonal projection is:

    .. math::

        P_{N_{r}} (\mathbf{X}) = \mathbf{U} diag(P_{L1_{r}}
                                 (\sigma(\mathbf{X}))) \mathbf{V}^H

    where :math:`\mathbf{U} diag(\sigma(\mathbf{X})) \mathbf{V}^H`
    is the SVD decomposition of :math:`\mathbf{X}`.

    """
    def __init__(self, n, radius, maxiter=100, xtol=1e-5):
        self.n = n
        self.radius = radius
        self.l1ball = L1BallProj(n, radius, maxiter, xtol)

    def __call__(self, X):
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        S = self.l1ball(S)
        return U @ np.diag(S) @ Vh