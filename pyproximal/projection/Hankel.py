import numpy as np
from scipy.linalg import hankel


class HankelProj:
    r"""Hankel matrix projection.

    Solves the least squares problem

        .. math::
          \min_{X\in\mathcal{H}} \|X-X_0\|_F^2

    where :math:`\mathcal{H}` is the set of Hankel matrices.

    Notes
    -----
    The solution to the above-mentioned least squares problem is given by a Hankel matrix,
    where the (constant) anti-diagonals are the average value along the corresponding
    anti-diagonals of the original matrix :math:`X_0`.
    """

    def __call__(self, X):
        m, n = X.shape
        ind = hankel(np.arange(m, dtype=np.int32), m - 1 + np.arange(n, dtype=np.int32))
        mean_values = np.bincount(ind.ravel(), weights=X.ravel()) / np.bincount(ind.ravel())
        return mean_values[ind]
