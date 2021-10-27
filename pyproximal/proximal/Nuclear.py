import numpy as np

from pylops.optimization.sparsity import _softthreshold
from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import NuclearBallProj
from pyproximal import ProxOperator


class Nuclear(ProxOperator):
    r"""Nuclear norm proximal operator.

    Proximal operator of the Nuclear norm defined as
    :math:`\sigma||\mathbf{X}||_* = \sigma \sum_i \lambda_i` where
    :math:`\mathbf{X}` is a matrix of size :math:`N \times M` and
    :math:`\lambda_i=1,2, min(N, M)` are its eigenvalues.

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`
    sigma : :obj:`int`, optional
        Multiplicative coefficient of Nuclear norm

    Notes
    -----
    The Nuclear norm proximal operator is defined as:

    .. math::

        prox_{\tau \sigma ||.||_*}(\mathbf{X}) =
        \mathbf{U} diag{prox_{\tau \sigma ||.||_1}(\boldsymbol\lambda)} \mathbf{V}

    where :math:`\mathbf{U}`, :math:`\boldsymbol\lambda`, and
    :math:`\mathbf{V}` define the SVD of :math:`X`.

    """
    def __init__(self, dim, sigma=1.):
        super().__init__(None, False)
        self.dim = dim
        self.sigma = sigma

    def __call__(self, x):
        X = x.reshape(self.dim)
        eigs = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0 # ensure all eigenvalues at positive
        nucl = np.sum(np.sqrt(eigs))
        return self.sigma * nucl

    @_check_tau
    def prox(self, x, tau):
        X = x.reshape(self.dim)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        Sth = _softthreshold(S, tau * self.sigma)
        X = np.dot(U * Sth, Vh)
        return X.ravel()


class NuclearBall(ProxOperator):
    r"""Nuclear ball proximal operator.

    Proximal operator of the Nuclear ball: :math:`N_{r} =
    \{ \mathbf{X}: ||\mathbf{X}||_* \leq r \}`.

    Parameters
    ----------
    dims : :obj:`tuple`
        Dimensions of input matrix
    radius : :obj:`float`
        Radius
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    As the Nuclear ball is an indicator function, the proximal operator
    corresponds to its orthogonal projection
    (see :class:`pyproximal.projection.NuclearBallProj` for details.

    """
    def __init__(self, dims, radius, maxiter=100, xtol=1e-5):
        super().__init__(None, False)
        self.dims = dims
        self.radius = radius
        self.maxiter = maxiter
        self.xtol = xtol
        self.ball = NuclearBallProj(min(self.dims), self.radius,
                                    self.maxiter, self.xtol)

    def __call__(self, x, tol=1e-5):
        return np.linalg.norm(x.reshape(self.dims), ord='nuc') - self.radius < tol

    @_check_tau
    def prox(self, x, tau):
        y = self.ball(x.reshape(self.dims))
        return y.ravel()
