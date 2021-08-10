import time
import numpy as np

from pylops import Gradient, BlockDiag
from pyproximal import Simplex, L1, L21, VStack
from pyproximal.optimization.primaldual import PrimalDual


def Segment(y, cl, sigma, alpha, clsigmas=None, z=None, niter=10, x0=None,
            callback=None, show=False, kwargs_simplex=None):
    r"""Primal-dual algorithm for image segmentation

    Perform image segmentation over :math:`N_{cl}` classes using the
    general version of the first-order primal-dual algorithm [1]_.

    Parameters
    ----------
    y : :obj:`np.ndarray`
        Image to segment (must have 2 or more dimensions)
    cl : :obj:`numpy.ndarray`
        Classes
    sigma : :obj:`float`
        Positive scalar weight of the misfit term
    alpha : :obj:`float`
        Positive scalar weight of the regularization term
    clsigmas : :obj:`numpy.ndarray`, optional
        Classes standard deviations
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    x0 : :obj:`numpy.ndarray`, optional
        Initial vector
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    kwargs_simplex : :obj:`dict`, optional
        Arbitrary keyword arguments for
        :py:func:`pyproximal.Simplex` operator

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Classes probabilities. This is a vector of size :math:`N_{dim} \times
        N_{cl}` whose columns contain the probability for each pixel to be in
        the class :math:`c_i`
    cl : :obj:`numpy.ndarray`
        Estimated classes. This is a vector of the same size of the input data
        ``y`` with the selected classes at each pixel.

    Notes
    -----
    This solver performs image segmentation over :math:`N_{cl}` classes solving
    the following nonlinear minimization problem using the general version of
    the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} \frac{\sigma}{2} \mathbf{x}^T \mathbf{f} +
        \mathbf{x}^T \mathbf{z} + \frac{\alpha}{2}||\nabla \mathbf{x}||_{2,1}

    where :math:`X=\{ \mathbf{x}: \sum_{i=1}^{N_{cl}} x_i = 1,\; x_i \geq 0 \}`
    is a simplex and :math:`\mathbf{f}=[\mathbf{f}_1, ...,
    \mathbf{f}_{N_{cl}}]^T` with :math:`\mathbf{f}_i = |\mathbf{y}-c_i|^2/\sigma_i`.
    Here :math:`\mathbf{c}=[c_1, ..., c_{N_{cl}}]^T` and
    :math:`\mathbf{\sigma}=[\sigma_1, ..., \sigma_{N_{cl}}]^T` are vectors
    representing the optimal mean and standard deviations for each class.

    .. [1] Chambolle, and A., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120–145. 2011.

    """
    kwargs_simplex = {} if kwargs_simplex is None else kwargs_simplex

    dims = y.shape
    ndims = len(dims)
    dimsprod = np.prod(np.array(dims))
    ncl = len(cl)

    # Data (difference between image and center of classes)
    g = sigma / 2. * (y.reshape(1, dimsprod) - cl[:, np.newaxis]) ** 2
    if clsigmas is not None:
        g /= clsigmas[:, np.newaxis]
    g = g.ravel()

    # Gradient operator
    sampling = 1.
    Gop = Gradient(dims=dims, sampling=sampling, edge=False,
                   kind='forward', dtype='float64')
    Gop = BlockDiag([Gop] * ncl)

    # Simplex and L1 proximal operators
    simp = Simplex(dimsprod * ncl, radius=1, dims=(ncl, dimsprod), axis=0,
                   **kwargs_simplex)
    #l1 = L1(sigma=0.5 * alpha)
    l21 = VStack([L21(ndim=ndims, sigma=0.5 * alpha)] * ncl,
                 nn=[ndims * dimsprod] * ncl)

    # Steps
    L = 8. / sampling ** 2
    tau = 1.
    mu = 1. / (tau * L)

    # Inversion
    x = PrimalDual(simp, l21, Gop, tau=tau, mu=mu,
                   z=g if z is None else g + z, theta=1.,
                   x0=np.zeros_like(g) if x0 is None else x0,
                   niter=niter, callback=callback, show=show)
    x = x.reshape(ncl, dimsprod).T
    cl = np.argmax(x, axis=1)
    cl = cl.reshape(dims)

    return x, cl
