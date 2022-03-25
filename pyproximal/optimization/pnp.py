import time
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.optimization.primal import ADMM


class _Denoise(ProxOperator):
    r"""Denoiser of choice

    Parameters
    ----------
    denoiser : :obj:`func`
        Denoiser (must be a function with two inputs, the first is the signal
        to be denoised, the second is the `tau` constant of the y-update in
        the ADMM optimization
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``prox`` method
        prior to calling the ``denoiser``

    """
    def __init__(self, denoiser, dims):
        super().__init__(None, False)
        self.denoiser = denoiser
        self.dims = dims

    def __call__(self, x):
        return 0.

    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.dims)
        xden = self.denoiser(x, tau)
        return xden.ravel()


def PlugAndPlay(proxf, denoiser, dims, x0, tau, niter=10,
                callback=None, show=False):
    r"""Plug-and-Play Priors with ADMM optimization

    Solves the following minimization problem using the ADMM algorithm:

    .. math::

        \mathbf{x},\mathbf{z}  = arg min_{\mathbf{x}}
        f(\mathbf{x}) + g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a function that has a known proximal
    operator where :math:`g(\mathbf{x})` is a function acting as implicit
    prior. Implicit means that no explicit function should be defined: instead,
    a denoising algorithm of choice is used. See Notes for details.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    denoiser : :obj:`func`
        Denoiser (must be a function with two inputs, the first is the signal
        to be denoised, the second is the tau constant of the y-update in
        PlugAndPlay)
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``prox`` method
        prior to calling the ``denoiser``
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    Notes
    -----
    Plug-and-Play Priors [1]_ can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
        \mathbf{z}^{k+1} = Denoise(\mathbf{x}^{k+1} + \mathbf{u}^{k}, \tau \sigma)\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

    where :math:`Denoise` is a denoising algorithm of choice and
    :math:`\tau \sigma` is the denoising parameter (should be chosen to
    represent an estimate of the noise variance). This rather peculiar step
    originates from the intuition that the optimization process associated
    with the z-update can be interpreted as a denoising inverse problem -
    for this reason any denoising of choice can be used instead of a function
    with known proximal operator.

    .. [1] Venkatakrishnan, S. V., Bouman, C. A. and Wohlberg, B.
       "Plug-and-Play priors for model based reconstruction",
       IEEE. 2013.

    """
    # Denoiser
    proxpnp = _Denoise(denoiser, dims=dims)

    return ADMM(proxf, proxpnp, tau=tau, x0=x0,
                niter=niter, callback=callback,
                show=show)