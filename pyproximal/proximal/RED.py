from collections.abc import Callable
from typing import Any

from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray, ShapeLike
from typing_extensions import Self

from pyproximal.proximal.L1 import _current_sigma
from pyproximal.ProxOperator import ProxOperator, _check_tau
from pyproximal.utils.typing import FloatCallableLike


class _Denoise:
    r"""Denoiser of choice

    Parameters
    ----------
    denoiser : :obj:`func`
        Denoiser (must be a function with two inputs, the first is the signal
        to be denoised, the second is the strenght of the denoiser `sigma`)
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``prox`` method
        prior to calling the ``denoiser``

    """

    def __init__(
        self,
        denoiser: Callable[[NDArray, float], NDArray],
        dims: ShapeLike,
    ) -> None:
        self.denoiser = denoiser
        self.dims = dims

    def __call__(self, x: NDArray, tau: float) -> NDArray:
        x = x.reshape(self.dims)
        xden = self.denoiser(x, tau)
        return xden.ravel()


class RED(ProxOperator):
    r"""Regularization by Denoising (RED)

    Regularization by Denoising loss:
    :math:`RED(\mathbf{x}) = \sigma\mathbf{x}^T (\mathbf{x} -
    f_{\sigma_d}(\mathbf{x}))`

    Parameters
    ----------
    denoiser : :obj:`func`
        Denoiser (must be a function with one input corresponding to
        the signal to be denoised)
    dims : :obj:`tuple`
        Dimensions used to reshape the vector ``x`` in the ``denoiser``
        method prior to applying the denoiser
    sigma : :obj:`float`, optional
        Multiplicative coefficient of RED term
    sigmad : :obj:`float` or :obj:`numpy.ndarray` or :obj:`func`, optional
        Strenght of the denoiser. This can be a constant number or a function
        that is called passing a counter which keeps track of how many
        times the ``grad`` or ``prox`` methods has been invoked before and
        returns a scalar (or a list of) ``sigma`` to be used
    x0 : :obj:`numpy.ndarray`, optional
        Initial vector of iterative scheme used to compute the proximal
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme used to compute the proximal
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    call : :obj:`bool`, optional
        Evalutate call method (``True``) or not (``False``)

    Notes
    -----
    The gradient of the RED loss is defined as:

    .. math::

        \nabla_\mathbf{x} RED(\mathbf{x}) =
        \sigma (\mathbf{x} - f_{\sigma_d}(\mathbf{x}))

    whilst the proximal operator is obtained by solving the
    minimization problem

    .. math::

        prox_{\tau RED} (\mathbf{x}) = \argmin_{\mathbf{y}} RED(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    via the following fixed-point iteration:

    .. math::

        \mathbf{y}^k = \frac{1}{\beta + \sigma} (\sigma f_{\sigma_d}(\mathbf{y}^{k-1})
        + \beta \mathbf{x})

    where :math:`\beta=1/\tau`.

    References
    ----------
    .. [1] Romano, Y., Elad, M., and Milanfar, P.
       "The Little Engine that Could Regularization by
       Denoising (RED)", SIAM Journal on Imaging Science.
       2017.

    """

    def __init__(
        self,
        denoiser: Callable[[NDArray, float], NDArray],
        dims: ShapeLike,
        sigma: float = 1.0,
        sigmad: FloatCallableLike = 1.0,
        x0: NDArray | None = None,
        niter: int = 10,
        warm: bool = True,
        call: bool = True,
    ) -> None:
        super().__init__(None, False)

        self.denoiser = _Denoise(denoiser, dims=dims)
        self.sigma = sigma
        self.sigmad = sigmad
        self.x0 = x0
        self.niter = niter
        self.warm = warm
        self.call = call
        self.count = 0

    def __call__(self, x: NDArray) -> bool | float:
        """Evaluate RED loss

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Vector

        Returns
        -------
        :obj:`float`
            - return ``0.0`` immediately if ``call=False``
            - return dot-product of the input and residual
              if ``call=True``
        """
        if not self.call:
            return 0.0
        else:
            ncp = get_array_module(x)
            sigmad = _current_sigma(self.sigmad, self.count)
            res = self.sigma * (x - self.denoiser(x, sigmad))
            return float(ncp.dot(x, res))

    def _increment_count(func: Callable[..., Any]) -> Callable[..., Any]:
        """Increment counter"""

        def wrapped(self: Self, *args: Any, **kwargs: Any) -> Any:
            self.count += 1
            return func(self, *args, **kwargs)

        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        ncp = get_array_module(x)
        beta = 1.0 / tau
        sigmad = _current_sigma(self.sigmad, self.count)

        # Define starting guess
        if self.x0 is None:
            sol = ncp.zeros_like(x)
        else:
            sol = self.x0

        # Fixed point iterations
        for _ in range(self.niter):
            den = self.denoiser(sol, sigmad)
            sol = (self.sigma * den + beta * x) / (self.sigma + beta)
        if self.warm:
            self.x0 = sol
        return sol

    @_increment_count
    def grad(self, x: NDArray) -> NDArray:
        sigmad = _current_sigma(self.sigmad, self.count)
        res = x - self.denoiser(x, sigmad)
        return self.sigma * res
