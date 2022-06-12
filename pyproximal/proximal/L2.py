import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import lsqr
from pylops import MatrixMult, Identity
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class L2(ProxOperator):
    r"""L2 Norm proximal operator.

    The Proximal operator of the :math:`\ell_2` norm is defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`
    and :math:`f_\alpha(\mathbf{x}) = f(\mathbf{x}) +
    \alpha \mathbf{q}^T\mathbf{x}`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    q : :obj:`numpy.ndarray`, optional
        Dot vector
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    alpha : :obj:`float`, optional
        Multiplicative coefficient of dot product
    qgrad : :obj:`bool`, optional
        Add q term to gradient (``True``) or not (``False``)
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    densesolver : :obj:`str`, optional
        Use ``numpy``, ``scipy``, or ``factorize`` when dealing with explicit
        operators. The former two rely on dense solvers from either library,
        whilst the last computes a factorization of the matrix to invert and
        avoids to do so unless the :math:`\tau` or :math:`\sigma` paramets
        have changed. Choose ``densesolver=None`` when using PyLops versions
        earlier than v1.18.1 or v2.0.0

    Notes
    -----
    The L2 proximal operator is defined as:

    .. math::

        prox_{\tau f_\alpha}(\mathbf{x}) =
        \left(\mathbf{I} + \tau \sigma \mathbf{Op}^T \mathbf{Op} \right)^{-1}
        \left( \mathbf{x} + \tau \sigma \mathbf{Op}^T \mathbf{b} -
        \tau \alpha \mathbf{q}\right)

    when both ``Op`` and ``b`` are provided. This formula shows that the
    proximal operator requires the solution of an inverse problem. If the
    operator ``Op`` is of kind ``explicit=True``, we can solve this problem
    directly. On the other hand if ``Op`` is of kind ``explicit=False``, an
    iterative solver is employed. In this case it is possible to provide a warm
    start via the ``x0`` input parameter.

    When only ``b`` is provided, ``Op`` is assumed to be an Identity operator
    and the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} + \tau \sigma \mathbf{b} - \tau \alpha \mathbf{q}}
        {1 + \tau \sigma}

    If ``b`` is not provided, the proximal operator reduces to:

    .. math::

        \prox_{\tau f_\alpha}(\mathbf{x}) =
        \frac{\mathbf{x} - \tau \alpha \mathbf{q}}{1 + \tau \sigma}

    Finally, note that the second term in :math:`f_\alpha(\mathbf{x})` is added
    because this combined expression appears in several problems where Bregman
    iterations are used alongside a proximal solver.

    """
    def __init__(self, Op=None, b=None, q=None, sigma=1., alpha=1.,
                 qgrad=True, niter=10, x0=None, warm=True, densesolver=None):
        super().__init__(Op, True)
        self.b = b
        self.q = q
        self.sigma = sigma
        self.alpha = alpha
        self.qgrad = qgrad
        self.niter = niter
        self.x0 = x0
        self.warm = warm
        self.densesolver = densesolver
        self.count = 0

        # when using factorize, store the first tau*sigma=0 so that the
        # first time it will be recomputed (as tau cannot be 0)
        if self.densesolver == 'factorize':
            self.tausigma = 0

        # create data term
        if self.Op is not None and self.b is not None:
            self.OpTb = self.sigma * self.Op.H @ self.b
            # create A.T A upfront for explicit operators
            if self.Op.explicit:
                self.ATA = np.conj(self.Op.A.T) @ self.Op.A

    def __call__(self, x):
        if self.Op is not None and self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(self.Op * x - self.b) ** 2)
        elif self.b is not None:
            f = (self.sigma / 2.) * (np.linalg.norm(x - self.b) ** 2)
        else:
            f = (self.sigma / 2.) * (np.linalg.norm(x) ** 2)
        if self.q is not None:
            f += self.alpha * np.dot(self.q, x)
        return f

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        # solve proximal optimization
        if self.Op is not None and self.b is not None:
            y = x + tau * self.OpTb
            if self.q is not None:
                y -= tau * self.alpha * self.q
            if self.Op.explicit:
                if self.densesolver != 'factorize':
                    Op1 = MatrixMult(np.eye(self.Op.shape[1]) +
                                     tau * self.sigma * self.ATA)
                    if self.densesolver is None:
                        # to allow backward compatibility with PyLops versions earlier
                        # than v1.18.1 and v2.0.0
                        x = Op1.div(y)
                    else:
                        x = Op1.div(y, densesolver=self.densesolver)
                else:
                    if self.tausigma != tau * self.sigma:
                        # recompute factorization
                        self.tausigma = tau * self.sigma
                        ATA = np.eye(self.Op.shape[1]) + \
                              self.tausigma * self.ATA
                        self.cl = cho_factor(ATA)
                    x = cho_solve(self.cl, y)
            else:
                Op1 = Identity(self.Op.shape[1], dtype=self.Op.dtype) + \
                      tau * self.sigma * self.Op.H * self.Op
                x = lsqr(Op1, y, iter_lim=niter, x0=self.x0)[0]
            if self.warm:
                self.x0 = x
        elif self.b is not None:
            num = x + tau * self.sigma * self.b
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        else:
            num = x
            if self.q is not None:
                num -= tau * self.alpha * self.q
            x = num / (1. + tau * self.sigma)
        return x

    def grad(self, x):
        if self.Op is not None and self.b is not None:
            g = self.sigma * self.Op.H @ (self.Op @ x - self.b)
        elif self.b is not None:
            g = self.sigma * (x - self.b)
        else:
            g = self.sigma * x
        if self.q is not None and self.qgrad:
            g += self.alpha * self.q
        return g


class L2Convolve(ProxOperator):
    r"""L2 Norm proximal operator with convolution operator

    Proximal operator for the L2 norm defined as: :math:`f(\mathbf{x}) =
    \frac{\sigma}{2} ||\mathbf{h} * \mathbf{x} - \mathbf{b}||_2^2` where
    :math:`\mathbf{h}` is the kernel of a convolution operator and
    :math:`*` represents convolution

    Parameters
    ----------
    h : :obj:`np.ndarray`, optional
        Kernel of convolution operator
    b : :obj:`numpy.ndarray`, optional
        Data vector
    b : :obj:`int`, optional
        Fourier transform number of samples
    sigma : :obj:`int`, optional
        Multiplicative coefficient of L2 norm
    dims : :obj:`tuple`, optional
        Number of samples for each dimension
        (``None`` if only one dimension is available)
    dir : :obj:`int`, optional
        Direction along which smoothing is applied.

    Notes
    -----
    The L2Convolve proximal operator is defined as:

    .. math::

        prox_{\tau f}(\mathbf{x}) =
        F^{-1}\left(\frac{\tau\sigma F(\mathbf{h})^* F(\mathbf{b}) + F(\mathbf{x})}
        {1 + \tau\sigma F(\mathbf{h})^* F(\mathbf{h})} \right)

    """
    def __init__(self, h, b=None, nfft=2**10, sigma=1., dims=None, dir=None):
        super().__init__(None, True)
        self.nfft = nfft
        self.sigma = sigma
        self.dims = dims
        self.dir = -1 if dir is None else dir

        # convert data and filter to Fourier domain
        self.bf = np.fft.fft(b, self.nfft, axis=self.dir)
        self.hf = np.fft.fft(h, self.nfft, axis=self.dir)

        # expand dimensions of filters
        if self.dims is not None:
            self.bf = self.bf.reshape(self.dims)
            self.dimsf = list(dims).copy()
            self.dimsf[dir] = nfft

            ndims = len(dims)
            for _ in range(dir - 1):
                self.hf = np.expand_dims(self.hf, axis=0)
            for _ in range(ndims - dir - 1):
                self.hf = np.expand_dims(self.hf, axis=-1)

        # precompute terms for prox
        self.hbf = np.conj(self.hf) * self.bf
        self.h2f = np.abs(self.hf) ** 2

    def __call__(self, x):
        if self.dims is not None:
            x = x.reshape(self.dims)
        xf = np.fft.fft(x, self.nfft, axis=self.dir)
        f = (self.sigma / 2.) * np.linalg.norm(np.fft.ifft(self.bf - self.hf * xf,
                                                           axis=self.dir)) ** 2
        return f

    @_check_tau
    def prox(self, x, tau):
        if self.dims is not None:
            x = x.reshape(self.dims)
        xf = np.fft.fft(x, self.nfft, axis=self.dir)
        yf = (xf + self.sigma * tau * self.hbf) / \
             (1. + self.sigma * tau * self.h2f)
        y = np.fft.ifft(yf, axis=self.dir)
        if self.dims is None:
            y = y[:len(x)]
        else:
            y = np.take(y, range(self.dims[self.dir]), axis=self.dir).ravel()
        return y.ravel()

    def grad(self, x):
        if self.dims is not None:
            x = x.reshape(self.dims)
        xf = np.fft.fft(x, self.nfft, axis=self.dir)

        f = self.sigma * np.fft.ifft(np.conj(self.hf) * (self.hf * xf - self.bf),
                                     axis=self.dir)
        return f.ravel()