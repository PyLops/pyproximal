import numpy as np
from pyproximal.projection import SimplexProj


class L1BallProj():
    r"""L1 ball projection.

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
    Given an L1 ball defined as:

    .. math::

        L1_{r} = \{ \mathbf{x}: ||\mathbf{x}||_1 \leq r \}

    its orthogonal projection is:

    .. math::

        P_{L1_{r}} (\mathbf{x}) = sign(x) P_{Simplex(r)}

    Note the this is the proximal operator of the corresponding
    indicator function :math:`I_{L1_{r}}`.

    """
    def __init__(self, n, radius, maxiter=100, xtol=1e-5):
        self.n = n
        self.radius = radius
        self.simplex = SimplexProj(n, radius, maxiter, xtol)

    def __call__(self, x):
        return np.sign(x) * self.simplex(np.abs(x))