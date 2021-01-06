import numpy as np
from pyproximal.projection import HyperPlaneBoxProj


class SimplexProj():
    r"""Simplex projection.

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
    Given a Simplex set defined as:

    .. math::

        \Delta_n(r) = \{ \mathbf{x}: \sum_i x_i = r,\; x_i \geq 0 \}

    its orthogonal projection is simply given by projection of the intersection
    between a hyperplane and a box with chosen radius and bounds equal to 0
    and inf (see :class:`pyproximal.projection.HyperPlaneBoxProj` for more
    details).

    """
    def __init__(self, n, radius, maxiter=100, xtol=1e-5):
        self.simplex = HyperPlaneBoxProj(np.ones(n), radius,
                                         lower=0, upper=np.inf,
                                         maxiter=maxiter, xtol=xtol)

    def __call__(self, x):
        """Apply Simplex projection

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        maxiter : :obj:`int`, optional
            Maximum number of iterations used by :func:`scipy.optimize.bisect`
        xtol : :obj:`float`, optional
            Absolute tolerance of :func:`scipy.optimize.bisect`

        """
        return self.simplex(x)