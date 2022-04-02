import numpy as np
from pyproximal.projection import HyperPlaneBoxProj


class IntersectionProj():
    r"""Intersection of multiple convex sets

    Parameters
    ----------
    k : :obj:`int`
        Size of vector to be projected
    n : :obj:`int`
        Number of vectors to be projected simultaneously
    sigma : :obj:`np.ndarray`
        Matrix of distances of size :math:`k \times k`
    k : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Tolerance of update

    Notes
    -----
    Given an Intersection of simple sets defined as:

    .. math::

        K = \bigcap_{1 \leq i_1 < i_2 \leq k} K_{i_1,i_2}, \quad
        K_{i_1,i_2}= \{ \mathbf{x}: |x_{i_2} - x_{i_1}| \leq \sigma_{i1, i2} \}

    its orthogonal projection can be obtained using the Dykstra's
    algorithm [1]_.

    .. [1] A., Chambolle, D., Cremers, and T., Pock, "A Convex Approach to
        Minimal Partitions", Journal of Mathematical, 2011.

    """
    def __init__(self, k, n, sigma, niter=100, tol=1e-5):
        self.k, self.n = k, n
        if isinstance(sigma, np.ndarray):
            self.sigma = sigma
        else:
            self.sigma = sigma * np.ones((k, k))
        self.niter = niter
        self.tol = tol

    def __call__(self, x):
        x = x.reshape(self.k, self.n)
        x12 = np.zeros((self.k, self.k, self.n))
        for iiter in range(self.niter):
            xold = x.copy()
            for i1 in range(self.k - 1):
                for i2 in range(i1 + 1, self.k):
                    xtilde = x[i2] - x[i1] + x12[i1, i2]
                    xtildeabs = np.abs(xtilde)
                    xdtilde = \
                        np.maximum(0, xtildeabs - self.sigma[i1, i2]) * \
                        xtilde / (xtildeabs + 1e-10)
                    x[i1] = x[i1] + 0.5 * (xdtilde - x12[i1, i2])
                    x[i2] = x[i2] - 0.5 * (xdtilde - x12[i1, i2])
                    x12[i1, i2] = xdtilde
            if max(np.sum(np.abs(x - xold), axis=0)) < self.tol:
                break
        return x.ravel()