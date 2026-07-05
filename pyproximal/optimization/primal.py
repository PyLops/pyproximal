__all__ = [
    "ProximalPoint",
    "ProximalGradient",
    "AcceleratedProximalGradient",
    "AndersonProximalGradient",
    "GeneralizedProximalGradient",
    "HQS",
    "ADMM",
    "ADMML2",
    "LinearizedADMM",
    "TwIST",
    "DouglasRachfordSplitting",
    "PPXA",
    "ConsensusADMM",
]

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pylops.optimization.callback import CostNanInfCallback, CostToInitialCallback
from pylops.utils.typing import NDArray

from pyproximal.optimization.cls_primal import ADMM as cADMM
from pyproximal.optimization.cls_primal import ADMML2 as cADMML2
from pyproximal.optimization.cls_primal import HQS as cHQS
from pyproximal.optimization.cls_primal import PPXA as cPPXA
from pyproximal.optimization.cls_primal import (
    AndersonProximalGradient as cAndersonProximalGradient,
)
from pyproximal.optimization.cls_primal import ConsensusADMM as cConsensusADMM
from pyproximal.optimization.cls_primal import (
    DouglasRachfordSplitting as cDouglasRachfordSplitting,
)
from pyproximal.optimization.cls_primal import (
    GeneralizedProximalGradient as cGeneralizedProximalGradient,
)
from pyproximal.optimization.cls_primal import LinearizedADMM as cLinearizedADMM
from pyproximal.optimization.cls_primal import ProximalGradient as cProximalGradient
from pyproximal.optimization.cls_primal import ProximalPoint as cProximalPoint
from pyproximal.optimization.cls_primal import TwIST as cTwIST

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator

    from pyproximal.ProxOperator import ProxOperator


def ProximalPoint(
    prox: "ProxOperator",
    x0: NDArray,
    tau: float,
    niter: int = 10,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Proximal point algorithm

    Solves the following minimization problem using Proximal point algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x})

    where :math:`f(\mathbf{x})` is any convex function that has a known
    proximal operator.

    Parameters
    ----------
    prox : :obj:`pyproximal.ProxOperator`
        Proximal operator
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight
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

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.ProximalPoint`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    proxpsolve = cProximalPoint(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        proxpsolve.callback = callback
    x, _, _ = proxpsolve.solve(
        prox=prox,
        x0=x0,
        tau=tau,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x


def ProximalGradient(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    epsg: float | NDArray = 1.0,
    tau: float | None = None,
    backtracking: bool = False,
    beta: float = 0.5,
    eta: float = 1.0,
    niter: int = 10,
    niterback: int = 100,
    acceleration: str | None = None,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Proximal gradient (optionally accelerated)

    Solves the following minimization problem using (Accelerated) Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor of g function. Can be a scalar
        for iteration-independent scaling or a a 1d vector for
        iteration-dependent scaling
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration. Finally, note that :math:`\tau` can be chosen to be a vector
        when dealing with problems with multiple right-hand-sides
    backtracking : :obj:`bool`, optional
        Force backtracking, even if ``tau`` is not equal to ``None``. In this case
        the chosen ``tau`` will be used as the initial guess in the first
        step of backtracking
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 1, 0 excluded).
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    acceleration : :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
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

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.ProximalGradient`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    proxgsolve = cProximalGradient(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        proxgsolve.callback = callback
    x, _, _, _ = proxgsolve.solve(
        proxf=proxf,
        proxg=proxg,
        x0=x0,
        epsg=epsg,
        tau=tau,
        backtracking=backtracking,
        beta=beta,
        eta=eta,
        acceleration=acceleration,
        niterback=niterback,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x


def AcceleratedProximalGradient(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    tau: float | None = None,
    beta: float = 0.5,
    epsg: float = 1.0,
    niter: int = 10,
    niterback: int = 100,
    acceleration: str = "vandenberghe",
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Accelerated Proximal gradient

    This is a thin wrapper around :func:`pyproximal.optimization.primal.ProximalGradient` with
    ``vandenberghe`` or ``fista`` acceleration. See :func:`pyproximal.optimization.primal.ProximalGradient`
    for details.

    """
    warnings.warn(
        "AcceleratedProximalGradient has been integrated directly into ProximalGradient "
        "from v0.5.0. It is recommended to start using ProximalGradient by selecting the "
        "appropriate acceleration parameter as this behaviour will become default in "
        "version v1.0.0 and AcceleratedProximalGradient will be removed.",
        FutureWarning,
        stacklevel=2,
    )
    return ProximalGradient(
        proxf,
        proxg,
        x0,
        tau=tau,
        beta=beta,
        epsg=epsg,
        niter=niter,
        niterback=niterback,
        acceleration=acceleration,
        tol=tol,
        rtol=rtol,
        callback=callback,
        show=show,
        itershow=itershow,
    )


def AndersonProximalGradient(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    epsg: float | NDArray = 1.0,
    tau: float | NDArray = 1.0,
    niter: int = 10,
    nhistory: int = 10,
    epsr: float = 1e-10,
    safeguard: bool = False,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Proximal gradient with Anderson acceleration

    Solves the following minimization problem using the Proximal
    gradient algorithm with Anderson acceleration:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor of g function. Can be a scalar
        for iteration-independent scaling or a a 1d vector for
        iteration-dependent scaling
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. N   ote that :math:`\tau`
        can be chosen to be a vector when dealing with problems with
        multiple right-hand-sides
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    nhistory : :obj:`int`, optional
        Number of previous iterates to be kept in memory (to compute the scaling factors
    epsr : :obj:`float`, optional
        Scaling factor for regularization added to the inverse of :math:\mathbf{R}^T \mathbf{R}`
    safeguard : :obj:`bool`, optional
        Apply safeguarding strategy to the update (``True``) or not (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
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

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.AndersonProximalGradient`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    aproxgsolve = cAndersonProximalGradient(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        aproxgsolve.callback = callback
    x, _, _, _ = aproxgsolve.solve(
        proxf=proxf,
        proxg=proxg,
        x0=x0,
        epsg=epsg,
        tau=tau,
        niter=niter,
        nhistory=nhistory,
        epsr=epsr,
        safeguard=safeguard,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x


def GeneralizedProximalGradient(
    proxfs: list["ProxOperator"],
    proxgs: list["ProxOperator"],
    x0: NDArray,
    tau: float | None,
    epsg: float | NDArray = 1.0,
    weights: NDArray | None = None,
    eta: float = 1.0,
    niter: int = 10,
    acceleration: str | None = None,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Generalized Proximal gradient

    Solves the following minimization problem using Generalized Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^n f_i(\mathbf{x})
        + \sum_{j=1}^m \epsilon_j g_j(\mathbf{x}),~~n,m \in \mathbb{N}^+

    where the :math:`f_i(\mathbf{x})` are smooth convex functions with a uniquely
    defined gradient and the :math:`g_j(\mathbf{x})` are any convex function that
    have a known proximal operator.

    Parameters
    ----------
    proxfs : :obj:`list`
        Proximal operators of the :math:`f_i` functions (must have ``grad`` implemented)
    proxgs : :obj:`list`
        Proximal operators of the :math:`g_j` functions
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\sum_{i=1}^n \nabla f_i`.
    epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
        Scaling factor(s) of ``g`` function(s). If a scalar is provided
        the same scaling factor is applied to every ``g`` function.
    weights : :obj:`float`, optional
        Weighting factors of ``g`` functions. Must sum to 1.
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 1, 0 excluded). Note that
        this will be only used when ``acceleration=None``.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    acceleration:  :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
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

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.GeneralizedProximalGradient`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    proxgsolve = cGeneralizedProximalGradient(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        proxgsolve.callback = callback
    x, _, _, _ = proxgsolve.solve(
        proxfs=proxfs,
        proxgs=proxgs,
        x0=x0,
        epsg=epsg,
        weights=weights,
        tau=1.0 if tau is None else tau,
        eta=eta,
        niter=niter,
        acceleration=acceleration,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x


def HQS(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    tau: float | NDArray,
    niter: int = 10,
    z0: NDArray | None = None,
    gfirst: bool = True,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    callbackz: bool = False,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> tuple[NDArray, NDArray]:
    r"""Half Quadratic splitting

    Solves the following minimization problem using Half Quadratic splitting
    algorithm:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector (not required when ``gfirst=False``, can pass ``None``)
    tau : :obj:`float` or :obj:`numpy.ndarray`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. Finally note that
        :math:`\tau` can be chosen to be a vector of size ``niter`` such that
        different :math:`\tau` is used at different iterations (i.e., continuation
        strategy)
    niter : :obj:`int`
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`, optional
        Initial z vector (not required when ``gfirst=True``)
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
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
    z : :obj:`numpy.ndarray`
        Inverted second model

    Raises
    ------
    ValueError
        If both ``x0`` and ``z0`` are set to ``None``

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.HQS`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    hqssolve = cHQS(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        hqssolve.callback = callback
    x, z, _, _ = hqssolve.solve(
        proxf=proxf,
        proxg=proxg,
        x0=x0,
        tau=tau,
        z0=z0,
        niter=niter,
        gfirst=gfirst,
        tol=tol or (0.0 if rtol else None),
        callbackz=callbackz,
        show=show,
        itershow=itershow,
    )
    return x, z


def ADMM(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    tau: float,
    niter: int = 10,
    z0: NDArray | None = None,
    gfirst: bool = False,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    callbackz: bool = False,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> tuple[NDArray, NDArray]:
    r"""Alternating Direction Method of Multipliers

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    ADMM can also solve the problem of the form above with a more general
    constraint: :math:`\mathbf{Ax}+\mathbf{Bz}=\mathbf{c}`. This routine implements
    the special case where :math:`\mathbf{A}=\mathbf{I}`, :math:`\mathbf{B}=-\mathbf{I}`,
    and :math:`\mathbf{c}=\mathbf{0}`, as a general algorithm can be obtained for any choice of
    :math:`f` and :math:`g` provided they have a known proximal operator.

    On the other hand, for more general choice of :math:`\mathbf{A}`, :math:`\mathbf{B}`,
    and :math:`\mathbf{c}`, the iterations are not generalizable, i.e. they depend on the choice of
    the :math:`f` and :math:`g` functions. For this reason, we currently only provide an additional
    solver for the special case where :math:`f` is a :class:`pyproximal.proximal.L2`
    operator with a linear operator  :math:`\mathbf{G}` and data :math:`\mathbf{y}`,
    :math:`\mathbf{B}=-\mathbf{I}` and :math:`\mathbf{c}=\mathbf{0}`,
    called :func:`pyproximal.optimization.primal.ADMML2`. Note that for the very same choice
    of :math:`\mathbf{B}` and :math:`\mathbf{c}`, the :func:`pyproximal.optimization.primal.LinearizedADMM`
    can also be used (and this does not require a specific choice of :math:`f`).

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector (not required when ``gfirst=False``, can pass ``None``)
    tau : :obj:`float`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`.
    niter : :obj:`int`
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`, optional
        Initial z vector (not required when ``gfirst=True``)
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
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
    z : :obj:`numpy.ndarray`
        Inverted second model

    See Also
    --------
    ADMML2: ADMM with L2 misfit function
    LinearizedADMM: Linearized ADMM

    Raises
    ------
    ValueError
        If both ``x0`` and ``z0`` are set to ``None``

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.ADMM`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    admmsolve = cADMM(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        admmsolve.callback = callback
    x, z, _, _ = admmsolve.solve(
        proxf=proxf,
        proxg=proxg,
        x0=x0,
        tau=tau,
        z0=z0,
        gfirst=gfirst,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        callbackz=callbackz,
        show=show,
        itershow=itershow,
    )
    return x, z


def ADMML2(
    proxg: "ProxOperator",
    Op: "LinearOperator",
    b: NDArray,
    A: "LinearOperator",
    x0: NDArray,
    tau: float,
    niter: int = 10,
    z0: NDArray | None = None,
    gfirst: bool = False,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    callbackz: bool = False,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
    **kwargs_solver: dict[str, Any],
) -> tuple[NDArray, NDArray]:
    r"""Alternating Direction Method of Multipliers for L2 misfit term

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        \frac{1}{2}||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 + g(\mathbf{z}) \\
        s.t. \; \mathbf{Ax}=\mathbf{z}

    where :math:`g(\mathbf{z})` is any convex function that has a known proximal operator.

    Parameters
    ----------
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    Op : :obj:`pylops.LinearOperator`
        Linear operator of data misfit term
    b : :obj:`numpy.ndarray`
        Data
    A : :obj:`pylops.LinearOperator`
        Linear operator of regularization term
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau \in (0, 1/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
        to solve the x-update

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    Raises
    ------
    ValueError
        If both ``x0`` and ``z0`` are set to ``None`` or ``x0`` is set to None

    See Also
    --------
    ADMM: ADMM
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.ADMML2`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    admml2solve = cADMML2(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        admml2solve.callback = callback
    x, z, _, _ = admml2solve.solve(
        proxg=proxg,
        Op=Op,
        b=b,
        A=A,
        x0=x0,
        tau=tau,
        z0=z0,
        gfirst=gfirst,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        callbackz=callbackz,
        show=show,
        itershow=itershow,
        **kwargs_solver,
    )
    return x, z


def LinearizedADMM(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    A: "LinearOperator",
    x0: NDArray,
    tau: float,
    mu: float,
    niter: int = 10,
    z0: NDArray | None = None,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> tuple[NDArray, NDArray]:
    r"""Linearized Alternating Direction Method of Multipliers

    Solves the following minimization problem using Linearized Alternating
    Direction Method of Multipliers:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + g(\mathbf{A}\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    function that has a known proximal operator and :math:`\mathbf{A}` is a
    linear operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following
        condition to guarantee convergence: :math:`\mu \in (0,
        \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    mu : :obj:`float`, optional
        Second positive scalar weight, which should satisfy the following
        condition to guarantees convergence: :math:`\mu \in (0,
        \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
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
    z : :obj:`numpy.ndarray`
        Inverted second model

    Raises
    ------
    ValueError
        If both ``x0`` and ``z0`` are set to ``None``

    See Also
    --------
    ADMM: ADMM
    ADMML2: ADMM with L2 misfit function

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.LinearizedADMM`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    ladmmsolve = cLinearizedADMM(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        ladmmsolve.callback = callback
    x, z, _, _ = ladmmsolve.solve(
        proxf=proxf,
        proxg=proxg,
        A=A,
        x0=x0,
        tau=tau,
        mu=mu,
        z0=z0,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x, z


def TwIST(
    proxg: "ProxOperator",
    A: "LinearOperator",
    b: NDArray,
    x0: NDArray,
    alpha: float | None = None,
    beta: float | None = None,
    eigs: tuple[float, float] | None = None,
    niter: int = 10,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
    returncost: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    r"""Two-step Iterative Shrinkage/Threshold

    Solves the following minimization problem using Two-step Iterative
    Shrinkage/Threshold:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \frac{1}{2}
        ||\mathbf{b} - \mathbf{Ax}||_2^2 + g(\mathbf{x})

    where :math:`\mathbf{A}` is a linear operator and :math:`g(\mathbf{x})`
    is any convex function that has a known proximal operator.

    Parameters
    ----------
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    A : :obj:`pylops.LinearOperator`
        Linear operator
    b : :obj:`numpy.ndarray`
        Data
    x0 : :obj:`numpy.ndarray`
        Initial vector
    alpha : :obj:`float`, optional
        Positive scalar weight (if ``None``, estimated based on the
        eigenvalues of :math:`\mathbf{A}`, see Notes for details)
    beta : :obj:`float`, optional
        Positive scalar weight (if ``None``, estimated based on the
        eigenvalues of :math:`\mathbf{A}`, see Notes for details)
    eigs : :obj:`tuple`, optional
        Largest and smallest eigenvalues of :math:`\mathbf{A}^H \mathbf{A}`.
        If passed, computes `alpha` and `beta` based on them.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    itershow : :obj:`tuple`, optional
        Display set log for the first N1 steps, last N2 steps,
        and every N3 steps in between where N1, N2, N3 are the
        three element of the list.
    returncost : :obj:`bool`, optional
        Return cost function

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    j : :obj:`numpy.ndarray`, optional
        Cost function

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.TwIST`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    twistsolve = cTwIST(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        twistsolve.callback = callback
    x, _, j = twistsolve.solve(
        proxg=proxg,
        A=A,
        b=b,
        x0=x0,
        alpha=alpha,
        beta=beta,
        eigs=eigs,
        niter=niter,
        tol=tol or (0.0 if rtol or returncost else None),
        show=show,
        itershow=itershow,
    )
    if returncost:
        return x, j
    else:
        return x


def DouglasRachfordSplitting(
    proxf: "ProxOperator",
    proxg: "ProxOperator",
    x0: NDArray,
    tau: float,
    eta: float = 1.0,
    niter: int = 10,
    gfirst: bool = True,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    callbacky: bool = False,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> tuple[NDArray, NDArray]:
    r"""Douglas-Rachford Splitting

    Solves the following minimization problem using Douglas-Rachford Splitting
    algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + g(\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    functions that has known proximal operators.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 2, 0 excluded).
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
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
        Modify callback signature to (``callback(x, y)``)
        when ``callbacky=True``
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
    y : :obj:`numpy.ndarray`
        Inverted second model

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.DouglasRachfordSplitting`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    drssolve = cDouglasRachfordSplitting(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        drssolve.callback = callback
    x, y, _, _ = drssolve.solve(
        proxf=proxf,
        proxg=proxg,
        x0=x0,
        tau=tau,
        eta=eta,
        gfirst=gfirst,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x, y


def PPXA(  # pylint: disable=invalid-name
    proxfs: list["ProxOperator"],
    x0: NDArray | list[NDArray],
    tau: float,
    eta: float = 1.0,
    weights: NDArray | list[float] | None = None,
    niter: int = 1000,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Parallel Proximal Algorithm (PPXA)

    Solves the following minimization problem using
    Parallel Proximal Algorithm (PPXA):

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^m f_i(\mathbf{x})

    where :math:`f_i(\mathbf{x})` are any convex
    functions that has known proximal operators.

    Parameters
    ----------
    proxfs : :obj:`list`
        A list of proximable functions :math:`f_1, \ldots, f_m`.
    x0 : :obj:`numpy.ndarray` or :obj:`list`
        Initial vector :math:`\mathbf{x}` for all :math:`f_i` if 1-dimensional array
        is provided, or initial vectors :math:`\mathbf{x}_{i}` for each :math:`f_i`
        for :math:`i=1,\ldots,m` if a :obj:`list` of 1-dimensional arrays or a 2-dimensional
        array of size ``(m, d)`` is provided, where ``d`` is the dimension of :math:`\mathbf{x}_{i}`.
    tau : :obj:`float`
        Positive scalar weight
    eta : :obj:`float`, optional
        Relaxation parameter (must be between 0 and 2, 0 excluded).
    weights : :obj:`numpy.ndarray` or :obj:`list` or :obj:`None`, optional
        Weights :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`,
        Defaults to None, which means :math:`w_1 = \cdots = w_m = \frac{1}{m}.`
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme.
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
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

    See Also
    --------
    ConsensusADMM: Consensus ADMM

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.PPXA`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    ppxasolve = cPPXA(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        ppxasolve.callback = callback
    x, _, _, _ = ppxasolve.solve(
        proxfs=proxfs,
        x0=x0,
        tau=tau,
        eta=eta,
        weights=weights,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x


def ConsensusADMM(  # pylint: disable=invalid-name
    proxfs: list["ProxOperator"],
    x0: NDArray,
    tau: float,
    niter: int = 1000,
    tol: float | None = None,
    rtol: float | None = None,
    callback: Callable[..., None] | None = None,
    show: bool = False,
    itershow: tuple[int, int, int] = (10, 10, 10),
) -> NDArray:
    r"""Consensus ADMM

    Solves the following global consensus problem using ADMM:

    .. math::

        \argmin_{\mathbf{x_1}, \mathbf{x_2}, \ldots, \mathbf{x_m}}
        \sum_{i=1}^m f_i(\mathbf{x}_i) \quad \text{s.t.}
        \quad \mathbf{x_1} = \mathbf{x_2} = \cdots = \mathbf{x_m}

    where :math:`f_i(\mathbf{x})` are any convex
    functions that has known proximal operators.

    Parameters
    ----------
    proxfs : :obj:`list`
        A list of proximable functions :math:`f_1, \ldots, f_m`.
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`
        Positive scalar weight
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme.
    tol : :obj:`float`, optional
        Tolerance on change of objective function (used as stopping criterion). If
        ``tol=None``, run until ``niter`` is reached
    rtol : :obj:`float`, optional
        Relative tolerance on objective function wrt initial value. Stops
        the solver when the ratio of the current objective function to the
        initial objective function is below this value. If ``rtol=None``,
        run until ``niter`` is reached or the other tolerance criterion is
        met
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

    See Also
    --------
    ADMM: Alternating Direction Method of Multipliers
    PPXA: Parallel Proximal Algorithm

    Notes
    -----
    See :class:`pyproximal.optimization.cls_primal.ConsensusADMM`

    """
    callbacks = []
    if tol is not None or rtol is not None:
        callbacks.append(CostNanInfCallback())
    if rtol is not None:
        callbacks.append(CostToInitialCallback(rtol))

    ccadmmsolve = cConsensusADMM(
        callbacks=callbacks if len(callbacks) > 0 else None,
    )
    if callback is not None:
        ccadmmsolve.callback = callback
    _, x, _, _, _ = ccadmmsolve.solve(
        proxfs=proxfs,
        x0=x0,
        tau=tau,
        niter=niter,
        tol=tol or (0.0 if rtol else None),
        show=show,
        itershow=itershow,
    )
    return x
