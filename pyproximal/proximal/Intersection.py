import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.projection.Intersection import IntersectionProj
from pyproximal.ProxOperator import ProxOperator, _check_tau


class Intersection(ProxOperator):
    r"""Intersection of multiple convex sets operator.

    Parameters
    ----------
    k : :obj:`int`
        Size of vector to be projected
    n : :obj:`int`
        Number of vectors to be projected simultaneously
    sigma : :obj:`np.ndarray` or :obj:`float`
        Matrix of distances of size :math:`k \times k` (or single value in the
        case of constant matrix)
    niter : :obj:`int`, optional
        Number of iterations
    tol : :obj:`float`, optional
        Toleance of update
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)

    Notes
    -----
    As the Intersection is an indicator function, the proximal operator
    corresponds to its orthogonal projection (see
    :class:`pyproximal.projection.IntersectionProj` for details.

    """

    def __init__(
        self,
        k: int,
        n: int,
        sigma: float | NDArray,
        niter: int = 100,
        tol: float = 1e-5,
        call: bool = True,
    ) -> None:
        super().__init__(None, False)
        self.k, self.n = k, n
        self.sigma = sigma if isinstance(sigma, np.ndarray) else sigma * np.ones((k, k))
        self.call = call
        self.ic = IntersectionProj(k, n, sigma, niter=niter, tol=tol)

    def __call__(self, x: NDArray, tol: float = 1e-8) -> bool:
        if not self.call:
            return False
        x = x.reshape(self.k, self.n)
        for i in range(self.n):
            for i1 in range(self.k - 1):
                for i2 in range(i1 + 1, self.k):
                    if np.abs(x[i1, i] - x[i2, i]) > self.sigma[i1, i2] + tol:
                        return False
        return True

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        return self.ic(x)
