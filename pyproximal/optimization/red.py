import time
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from pylops import LinearOperator
from pylops.basicoperators import Identity
from pylops.optimization.basic import lsqr
from pylops.utils.backend import to_numpy
from pylops.utils.typing import NDArray

from pyproximal.optimization.primal import ADMM
from pyproximal.proximal import L2
from pyproximal.proximal import RED as pRED
from pyproximal.ProxOperator import ProxOperator


def _GradientDescent(
    f: ProxOperator,
    g: ProxOperator,
    x0: NDArray,
    alpha: float = 1.0,
    niter: int = 100,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
) -> NDArray:
    """Gradient descent solver for composite objective.

    Parameters
    ----------
    f: :obj:`pyproximal.ProxOperator`
        First proximal operator (must implement ``grad`` and ``call``)
    g: :obj:`pyproximal.ProxOperator`
        Second proximal operator (must implement ``grad`` and ``call``)
    x0 : :obj:`numpy.ndarray`
        Initial vector
    alpha : :obj:`float`, optional
        Step size
    niter : :obj:`int`, optional
        Number of iterations
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show: :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    """
    if show:
        tstart = time.time()
        print(
            "Gradient descent algorithm\n"
            "---------------------------------------------------------\n"
            "Proximal operator (f): %s\n"
            "Proximal operator (g): %s\n"
            "alpha = %10e\tniter = %d\n" % (type(f), type(g), alpha, niter)
        )
        head = "   Itn       x[0]          f"
        print(head)

    x = x0.copy()
    for iiter in range(niter):
        grad = f.grad(x).real + g.grad(x)
        x -= alpha * grad
        if callback is not None:
            callback(x)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = "%6g  %12.5e  %10.3e" % (
                    iiter + 1,
                    np.real(to_numpy(x[0])),
                    f(x),
                )
                print(msg)
    if show:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
    return x


def _FixedPoint(
    Op: LinearOperator,
    y: NDArray,
    denoiser: Callable[[NDArray, float], NDArray],
    x0: NDArray,
    sigmad: float | Callable[[int], Any],
    sigmaOp: float,
    sigma: float,
    niter: int = 100,
    niter_inner: int = 10,
    callback: Callable[[NDArray], None] | None = None,
    show: bool = False,
) -> NDArray:
    """Fixed-point solver for L2 data-misfit term and RED term.

    Parameters
    ----------
    Op: :obj:`pylops.LinearOperator`
        Linear operator of L2 data-misfit term
    y: :obj:`numpy.ndarray`
        Data vector
    denoiser: :obj:`callable`
        Denoising function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    sigmad : :obj:`float`, optional
        Strenght of the denoiser
    sigmaOp : :obj:`float`, optional
        Multiplicative coefficient of L2 data-misfit term
    sigma : :obj:`float` or :obj:`func`, optional
        Multiplicative coefficient of RED term
    niter : :obj:`int`, optional
        Number of iterations
    niter_inner : :obj:`int`, optional
        Number of iterations for the inner solver
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show: :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    """
    if show:
        tstart = time.time()
        print(
            "Fixed point algorithm\n"
            "---------------------------------------------------------\n"
            "Linear Operator: %s\n"
            "Denoiser: %s\n"
            "sigmad = %s\tsigmaOp = %10e\tsigma = %10e\n"
            "niter = %10e\tniter_inner = %d\n"
            % (
                type(Op),
                type(denoiser),
                "multi" if callable(sigmad) else str(sigmad),
                sigmaOp,
                sigma,
                niter,
                niter_inner,
            )
        )
        head = "   Itn       x[0]          f"
        print(head)

    x = x0.copy()
    yy = sigmaOp * Op.H @ y
    Iop = Identity(Op.shape[1], dtype=Op.dtype)
    Op1 = sigma * Iop + sigmaOp * Op.H * Op
    for iiter in range(niter):
        xden = denoiser(x, sigmad(iiter) if callable(sigmad) else sigmad)
        y1 = yy + sigma * xden
        x = x = lsqr(Op1, y1, niter=niter_inner, x0=x)[0]
        if callback is not None:
            callback(x)
            if show:
                if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                    msg = "%6g  %12.5e  %10.3e" % (
                        iiter + 1,
                        np.real(to_numpy(x[0])),
                        sigmaOp / 2.0 * np.linalg.norm(Op * x - y) ** 2,
                    )
                    print(msg)
    if show:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")

    return x


def RED(
    proxf: ProxOperator,
    red: pRED,
    x0: NDArray,
    solver: Callable[..., NDArray] | Literal["gradientdescent", "fixedpoint"] = ADMM,
    **kwargs_solver: dict[str, Any],
) -> NDArray | tuple[NDArray, ...]:
    r"""Regularization by Denoising (RED) solver

    Solves the following minimization problem:

    .. math::

        \mathbf{x}  = \argmin_{\mathbf{x}}
        f(\mathbf{x}) + \sigma \mathbf{x}^T (\mathbf{x} -
        g_{\sigma_d}(\mathbf{x}))

    using either any of the proximal solvers in
    :mod:`pyproximal.optimization.primal` or
    :mod:`pyproximal.optimization.primaldual`, or the
    gradient descent of fixed point solvers.

    Here :math:`f(\mathbf{x})` is a function that has a known gradient or
    proximal operator, whilst :math:`g_{\sigma_d}(\mathbf{x})` is a denoiser
    acting on a noisy signal with noise strenght equal to `\sigma_d`.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function. In the case of `solver="gradientdescent", it must implement
        the `grad` method
    red : :obj:`pyproximal.proximal.RED`
        RED operator
    x0 : :obj:`numpy.ndarray`
        Initial vector
    solver : :func:`pyproximal.optimization.primal` or :func:`pyproximal.optimization.primaldual` or :obj:`str`
        Solver of choice or ``"gradientdescent"`` for gradient-descent solver or ``"fixedpoint"``
        for fixed-point solver
    kwargs_solver : :obj:`dict`
        Additional parameters required by the selected solver. For ``solver="gradientdescent"``,
        the parameter  ``alpha`` corresponds to the step size; for ``solver="fixedpoint"``,
        the parameter ``niter_inner`` corresponds to the number of iterations of the inner loop

    Returns
    -------
    out : :obj:`numpy.ndarray` or :obj:`tuple`
        Output of the solver of choice. For

    Raises
    ------
    ValueError
        If ``solver="gradientdescent"`` and ``proxf`` does not implement the ``grad`` method
        or ``solver="fixedpoint"`` and ``proxf`` is not of :class:`pyproximal.proximal.L2` type.

    Notes
    -----
    The Regularization by Denoising (RED) term can be used with any proximal
    solvers, since its proximal operator can be easily evaluated via
    fixed-point iterations (see Notes of :class:`pyproximal.proximal.RED`
    for more details).

    However, [1]_ presented two additional solvers, namely:

    - gradient descent (for differentiable :math:`f`):

      .. math::

          \mathbf{x}^{k+1} = \mathbf{x}^k - \alpha
          (\nabla_f(\mathbf{x}^k) + \mathbf{x}^k -
          g_{\sigma_d}(\mathbf{x}^k))

    - fixed point (for :math:`f= \frac{\sigma_{\mathbf{Op}}}{2} ||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2`):

      .. math::

          \mathbf{y}^k = g_{\sigma_d}(\mathbf{x}^k)\\
          \mathbf{x}^{k+1} = (\sigma_{\mathbf{Op}} \mathbf{Op}^H\mathbf{Op} +
          \sigma \mathbf{I})^{-1} (\sigma_{\mathbf{Op}} \mathbf{Op}^H \mathbf{b} +
          \sigma \mathbf{y}^k)

    References
    ----------
    .. [1] Romano, Y., Elad, M., and Milanfar, P.
       "The Little Engine that Could Regularization by
       Denoising (RED)", SIAM Journal on Imaging Science.
       2017.

    """

    if isinstance(solver, str):
        if solver == "gradientdescent":
            if hasattr(proxf, "grad") and callable(proxf.grad):
                return _GradientDescent(
                    proxf, red, x0=x0, **cast(dict[str, Any], kwargs_solver)
                )
            else:
                msg = f"The provided proximal operator ({proxf}) does not implement the grad method..."
                raise ValueError(msg)
        else:
            if isinstance(proxf, L2):
                return _FixedPoint(
                    proxf.Op,
                    proxf.b,
                    red.denoiser,
                    x0,
                    red.sigmad,
                    proxf.sigma,
                    red.sigma,
                    **cast(dict[str, Any], kwargs_solver),
                )
            else:
                msg = f"The proximal operator must be of type L2, ({type(proxf).__name__}) provided instead..."
                raise ValueError(msg)
    else:
        return solver(proxf, red, x0=x0, **kwargs_solver)
