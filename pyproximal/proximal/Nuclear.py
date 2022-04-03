import numpy as np

from pylops.optimization.sparsity import _softthreshold
from pyproximal.ProxOperator import _check_tau
from pyproximal.projection import NuclearBallProj
from pyproximal import ProxOperator


class Nuclear(ProxOperator):
    r"""Nuclear norm proximal operator.

    The nuclear norm is defined as
    :math:`\sigma\|\mathbf{X}\|_* = \sigma \sum_i \lambda_i` where :math:`\mathbf{X}`
    is a matrix of size :math:`M \times N` and :math:`\lambda_i` is the *i*:th
    singular value of :math:`\mathbf{X}`, where :math:`i=1,\ldots, \min(M, N)`.

    The *weighted* nuclear norm, with the positive weight vector :math:`\boldsymbol\sigma`, is
    defined as

    .. math::

         \|\mathbf{X}|\|_{{\boldsymbol\sigma},*} = \sum_i \sigma_i\lambda_i(\mathbf{X}) .

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of matrix :math:`\mathbf{X}`
    sigma : :obj:`float` or :obj:`numpy.ndarray`, optional
        Multiplicative coefficient of the nuclear norm penalty. If ``sigma`` is a float
        the same penalty is applied for all singular values. If instead ``sigma`` is an
        array the weight ``sigma[i]`` will be applied to the *i*:th singular value.
        This is often referred to as the *weighted nuclear norm*.

    Notes
    -----
    The nuclear norm proximal operator is:

    .. math::

        \prox_{\tau \sigma \|\cdot\|_*}(\mathbf{X}) =
        \mathbf{U} \diag \{ \prox_{\tau \sigma \|\cdot\|_1}(\boldsymbol\lambda) \} \mathbf{V}^H

    where :math:`\mathbf{U}`, :math:`\boldsymbol\lambda`, and
    :math:`\mathbf{V}` define the SVD of :math:`X`.

    The weighted nuclear norm is convex if the sequence :math:`\{\sigma_i\}_i` is
    non-ascending, but is in general non-convex; however, when the weights are
    non-descending it can be shown that applying the soft-thresholding operator on the
    singular values still yields a fixed point (w. r. t. a specific algorithm), see
    [1]_ for details.

    .. [1] Gu et al. "Weighted Nuclear Norm Minimization with Application to Image
        Denoising", In the IEEE Conference on Computer Vision and Pattern Recognition,
        2862-2869, 2014.

    """

    def __init__(self, dim, sigma=1.):
        super().__init__(None, False)
        self.dim = dim
        self.sigma = sigma

    def __call__(self, x):
        X = x.reshape(self.dim)
        eigs = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0  # ensure all eigenvalues at positive
        return np.sum(np.flip(self.sigma) * np.sqrt(eigs))

    @_check_tau
    def prox(self, x, tau):
        X = x.reshape(self.dim)
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        sigma = self.sigma if np.isscalar(self.sigma) else self.sigma[:S.size]
        Sth = _softthreshold(S, tau * sigma)
        X = np.dot(U * Sth, Vh)
        return X.ravel()


class NuclearBall(ProxOperator):
    r"""Nuclear ball proximal operator.

    Proximal operator of the Nuclear ball: :math:`N_{r} =
    \{ \mathbf{X}: \|\mathbf{X}\|_* \leq r \}`.

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
