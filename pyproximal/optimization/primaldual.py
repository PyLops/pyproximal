__all__ = [
    "PrimalDual",
    "AdaptivePrimalDual",
]

from collections.abc import Callable
from typing import TYPE_CHECKING

from pylops.optimization.callback import CostNanInfCallback, CostToInitialCallback
from pylops.utils.typing import NDArray

from pyproximal.optimization.cls_primaldual import (
    AdaptivePrimalDual as cAdaptivePrimalDual,
)
from pyproximal.optimization.cls_primaldual import PrimalDual as cPrimalDual

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator

    from pyproximal.ProxOperator import ProxOperator


def PrimalDual(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    A: "LinearOperator",
    x0: NDArray,
    tau: float | NDArray,
    mu: float | NDArray,
    y0: NDArray | None = None,
    z: NDArray | None = None,
    theta: float = 1.0,
    niter: int = 10,
    gfirst: bool = True,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    callbacky: bool = False,
    returny: bool = False,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Primal-dual algorithm

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`numpy.ndarray`
        Stepsize of subgradient of :math:`f`. This can be constant
        or function of iterations (in the latter cases provided
        as numpy.ndarray)
    mu : :obj:`float` or :obj:`numpy.ndarray`
        Stepsize of subgradient of :math:`g^*`. This can be constant
        or function of iterations (in the latter cases provided as
        numpy.ndarray)
    y0 : :obj:`numpy.ndarray`
        Initial auxiliary vector. If ``None``, set to zero
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    theta : :obj:`float`
        Scalar between 0 and 1 that defines the update of the
        :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
        special case that represents the semi-implicit classical Arrow-Hurwicz
        algorithm
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on x/y updates (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached or the other tolerance
        criterion is met
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbacky : :obj:`bool`, optional
        Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
    returny : :obj:`bool`, optional
        Return also ``y``
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    y : :obj:`numpy.ndarray`, optional
        Inverted second model, only returned if ``returny=True``

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primaldual.PrimalDual`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    pdsolve = cPrimalDual(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        pdsolve.callback = callback
    x, _, y, _, _ = pdsolve.solve(
        proxf=proxf,
        proxg=proxg,
        A=A,
        x0=x0,
        tau=tau,
        mu=mu,
        y0=y0,
        z=z,
        theta=theta,
        gfirst=gfirst,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        callbacky=callbacky,
        show=show,
        itershow=itershow,
    )
    if not returny:
        return x
    else:
        return x, y


def AdaptivePrimalDual(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    A: "LinearOperator",
    x0: NDArray,
    tau: float,
    mu: float,
    alpha: float = 0.5,
    eta: float = 0.95,
    s: float = 1.0,
    delta: float = 1.5,
    z: NDArray | None = None,
    niter: int = 10,
    tol: float | None = None,
    rtol: float | None = None,
    xytol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> tuple[NDArray, tuple[NDArray, NDArray, NDArray]]:
    r"""Adaptive Primal-dual algorithm

    Solves the minimization problem in
    :func:`pyproximal.optimization.primaldual.PrimalDual`
    using an adaptive version of the first-order primal-dual algorithm.
    The main advantage of this method is that step sizes :math:`\tau` and
    :math:`\mu` are changing through iterations, improving the overall speed
    of convergence of the algorithm.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator of g
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Stepsize of subgradient of :math:`f`
    mu : :obj:`float`
        Stepsize of subgradient of :math:`g^*`
    alpha : :obj:`float`, optional
        Initial adaptivity level (must be between 0 and 1)
    eta : :obj:`float`, optional
        Scaling of adaptivity level to be multipled to the current alpha every
        time the norm of the two residuals start to diverge (must be between
        0 and 1)
    s : :obj:`float`, optional
        Scaling of residual balancing principle
    delta : :obj:`float`, optional
        Balancing factor. Step sizes are updated only when their ratio exceeds
        this value.
    z : :obj:`numpy.ndarray`, optional
        Additional vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached or the other tolerance
        criterion is met
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    xytol : :obj:`float`, optional
        Tolerance on x/y updates (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    steps : :obj:`tuple`
        Tau, mu and alpha evolution through iterations

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primaldual.AdaptivePrimalDual`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    apdsolve = cAdaptivePrimalDual(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        apdsolve.callback = callback
    x, _, _, _, steps = apdsolve.solve(
        proxf=proxf,
        proxg=proxg,
        A=A,
        x0=x0,
        tau=tau,
        mu=mu,
        alpha=alpha,
        eta=eta,
        s=s,
        delta=delta,
        z=z,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        xytol=xytol,
        show=show,
        itershow=itershow,
    )
    return x, steps
