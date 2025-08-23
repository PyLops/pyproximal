import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator
from pyproximal.utils.bilinear import BilinearOperator


def _backtracking(
    x: List[NDArray],
    tau: float,
    H: BilinearOperator,
    proxf: Optional[ProxOperator],
    ix: int,
    beta: float = 0.5,
    niterback: int = 10,
) -> Tuple[NDArray, float]:
    r"""Backtracking

    Line-search algorithm for finding step sizes in PALM algorithms when
    the Lipschitz constant of the operator is unknown (or expensive to
    estimate).

    """

    def ftilde(
        x: NDArray,
        y: List[NDArray],
        f: BilinearOperator,
        g: NDArray,
        tau: float,
        ix: int,
    ) -> float:
        xy = x - y[ix]
        return float(
            f(*y) + np.dot(g, xy) + (1.0 / (2.0 * tau)) * np.linalg.norm(xy) ** 2
        )

    iiterback = 0
    if ix == 0:
        grad = H.gradx(x[ix])
    else:
        grad = H.grady(x[ix])
    z = [x_.copy() for x_ in x]
    while iiterback < niterback:
        z[ix] = x[ix] - tau * grad
        if proxf is not None:
            z[ix] = proxf.prox(z[ix], tau)
        ft = ftilde(z[ix], x, H, grad, tau, ix)
        f = H(*z)
        if f <= ft or tau < 1e-12:
            break
        tau *= beta
        iiterback += 1
    return z[ix], tau


def PALM(
    H: BilinearOperator,
    proxf: Optional[ProxOperator],
    proxg: Optional[ProxOperator],
    x0: NDArray,
    y0: NDArray,
    gammaf: Optional[float] = 1.0,
    gammag: Optional[float] = 1.0,
    beta: float = 0.5,
    niter: int = 10,
    niterback: int = 100,
    callback: Optional[Callable[[NDArray, NDArray], None]] = None,
    show: bool = False,
) -> Tuple[NDArray, NDArray]:
    r"""Proximal Alternating Linearized Minimization

    Solves the following minimization problem using the Proximal Alternating
    Linearized Minimization (PALM) algorithm:

    .. math::

        \mathbf{x}\mathbf{,y} = \argmin_{\mathbf{x}, \mathbf{y}}
        f(\mathbf{x}) + g(\mathbf{y}) + H(\mathbf{x}, \mathbf{y})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{y})` are any pair of
    convex functions that have known proximal operators, and
    :math:`H(\mathbf{x}, \mathbf{y})` is a smooth function.

    Parameters
    ----------
    H : :obj:`pyproximal.utils.bilinear.Bilinear`
        Bilinear function
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial x vector
    y0 : :obj:`numpy.ndarray`
        Initial y vector
    gammaf : :obj:`float`, optional
        Positive scalar weight for ``f`` function update.
        If ``None``, use backtracking
    gammag : :obj:`float`, optional
        Positive scalar weight for ``g`` function update.
        If ``None``, use backtracking
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` and ``y`` are the current model vectors
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted x vector
    y : :obj:`numpy.ndarray`
        Inverted y vector

    Notes
    -----
    PALM [1]_ can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{c_k f}(\mathbf{x}^{k} -
        \frac{1}{c_k}\nabla_x H(\mathbf{x}^{k}, \mathbf{y}^{k}))\\
        \mathbf{y}^{k+1} = \prox_{d_k g}(\mathbf{y}^{k} -
        \frac{1}{d_k}\nabla_y H(\mathbf{x}^{k+1}, \mathbf{y}^{k}))\\

    Here :math:`c_k=\gamma_f L_x` and :math:`d_k=\gamma_g L_y`, where
    :math:`L_x` and :math:`L_y` are the Lipschitz constant of :math:`\nabla_x H`
    and :math:`\nabla_y H`, respectively. When such constants cannot be easily
    computed, a back-tracking algorithm can be instead employed to find suitable
    :math:`c_k` and :math:`d_k` parameters.

    .. [1] Bolte, J., Sabach, S., and Teboulle, M. "Proximal alternating
       linearized minimization for nonconvex and nonsmooth problems",
       Mathematical Programming, vol. 146, pp. 459–494. 2014.

    """
    if show:
        tstart = time.time()
        print(
            "PALM algorithm\n"
            "---------------------------------------------------------\n"
            "Bilinear operator: %s\n"
            "Proximal operator (f): %s\n"
            "Proximal operator (g): %s\n"
            "gammaf = %s\tgammag = %s\tniter = %d\n"
            % (type(H), type(proxf), type(proxg), str(gammaf), str(gammag), niter)
        )
        head = "   Itn      x[0]       y[0]        f         g         H         ck         dk"
        print(head)

    if gammaf is None:
        tauf = 1.0
        ck = 0.0
    if gammag is None:
        taug = 1.0
        dk = 0.0

    x, y = x0.copy(), y0.copy()
    for iiter in range(niter):
        # x step
        if gammaf is not None:
            ck = gammaf * H.ly(y)
            x = x - (1.0 / ck) * H.gradx(x)
            if proxf is not None:
                x = proxf.prox(x, 1.0 / ck)
        else:
            x, tauf = _backtracking(
                [x, y], tauf, H, proxf, 0, beta=beta, niterback=niterback
            )
        # update x parameter in H function
        H.updatex(x.copy())

        # y step
        if gammag is not None:
            dk = gammag * H.lx(x)
            y = y - (1.0 / dk) * H.grady(y)
            if proxg is not None:
                y = proxg.prox(y, 1.0 / dk)
        else:
            y, taug = _backtracking(
                [x, y], taug, H, proxf, 1, beta=beta, niterback=niterback
            )
        # update y parameter in H function
        H.updatey(y.copy())

        # run callback
        if callback is not None:
            callback(x, y)

        if show:
            pf = proxf(x) if proxf is not None else 0.0
            pg = proxg(y) if proxg is not None else 0.0
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = "%6g  %5.5e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e" % (
                    iiter + 1,
                    x[0],
                    y[0],
                    pf,
                    pg,
                    H(x, y),
                    ck,
                    dk,
                )
                print(msg)
    if show:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
    return x, y


def iPALM(
    H: BilinearOperator,
    proxf: Optional[ProxOperator],
    proxg: Optional[ProxOperator],
    x0: NDArray,
    y0: NDArray,
    gammaf: Optional[float] = 1.0,
    gammag: Optional[float] = 1.0,
    a: List[float] = [1.0, 1.0],
    beta: float = 0.5,
    niter: int = 10,
    niterback: int = 100,
    callback: Optional[Callable[[NDArray, NDArray], None]] = None,
    show: bool = False,
) -> Tuple[NDArray, NDArray]:
    r"""Inertial Proximal Alternating Linearized Minimization

    Solves the following minimization problem using the Inertial Proximal
    Alternating Linearized Minimization (iPALM) algorithm:

    .. math::

        \mathbf{x}\mathbf{,y} = \argmin_{\mathbf{x}, \mathbf{y}}
        f(\mathbf{x}) + g(\mathbf{y}) + H(\mathbf{x}, \mathbf{y})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{y})` are any pair of
    convex functions that have known proximal operators, and
    :math:`H(\mathbf{x}, \mathbf{y})` is a smooth function.

    Parameters
    ----------
    H : :obj:`pyproximal.utils.bilinear.Bilinear`
        Bilinear function
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial x vector
    y0 : :obj:`numpy.ndarray`
        Initial y vector
    gammaf : :obj:`float`, optional
        Positive scalar weight for ``f`` function update.
        If ``None``, use backtracking
    gammag : :obj:`float`, optional
        Positive scalar weight for ``g`` function update.
        If ``None``, use backtracking
    a : :obj:`list`, optional
        Inertial parameters (:math:`a  \in [0, 1]`)
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` and ``y`` are the current model vectors
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted x vector
    y : :obj:`numpy.ndarray`
        Inverted y vector

    Notes
    -----
    iPALM [1]_ can be expressed by the following recursion:

    .. math::

        \mathbf{x}_z^k = \mathbf{x}^k + \alpha_x (\mathbf{x}^k - \mathbf{x}^{k-1})\\
        \mathbf{x}^{k+1} = \prox_{c_k f}(\mathbf{x}_z^k  -
        \frac{1}{c_k}\nabla_x H(\mathbf{x}_z^k, \mathbf{y}^{k}))\\
        \mathbf{y}_z^k = \mathbf{y}^k + \alpha_y (\mathbf{y}^k - \mathbf{y}^{k-1})\\
        \mathbf{y}^{k+1} = \prox_{d_k g}(\mathbf{y}_z^k -
        \frac{1}{d_k}\nabla_y H(\mathbf{x}^{k+1}, \mathbf{y}_z^k))

    Here :math:`c_k=\gamma_f L_x` and :math:`d_k=\gamma_g L_y`, where
    :math:`L_x` and :math:`L_y` are the Lipschitz constant of :math:`\nabla_x H`
    and :math:`\nabla_y H`, respectively. When such constants cannot be easily
    computed, a back-tracking algorithm can be instead employed to find suitable
    :math:`c_k` and :math:`d_k` parameters.

    Finally, note that we have implemented the version of iPALM where :math:`\beta_x=\alpha_x`
    and :math:`\beta_y=\alpha_y`.

    .. [1] Pock, T., and Sabach, S. "Inertial Proximal
       Alternating Linearized Minimization (iPALM) for Nonconvex and
       Nonsmooth Problems", SIAM Journal on Imaging Sciences, vol. 9. 2016.

    """
    if show:
        tstart = time.time()
        print(
            "iPALM algorithm\n"
            "---------------------------------------------------------\n"
            "Bilinear operator: %s\n"
            "Proximal operator (f): %s\n"
            "Proximal operator (g): %s\n"
            "gammaf = %s\tgammag = %s\n"
            "a = %s\tniter = %d\n"
            % (
                type(H),
                type(proxf),
                type(proxg),
                str(gammaf),
                str(gammag),
                str(a),
                niter,
            )
        )
        head = "   Itn      x[0]       y[0]        f         g         H         ck         dk"
        print(head)

    if gammaf is None:
        tauf = 1.0
        ck = 0.0
    if gammaf is None:
        taug = 1.0
        dk = 0.0

    x, y = x0.copy(), y0.copy()
    xold, yold = x0.copy(), y0.copy()
    for iiter in range(niter):
        # x step
        z = x + a[0] * (x - xold)
        if gammaf is not None:
            ck = gammaf * H.ly(y)
            xold = x.copy()
            x = z - (1.0 / ck) * H.gradx(z)
            if proxf is not None:
                x = proxf.prox(x, 1.0 / ck)
        else:
            xold = x.copy()
            x, tauf = _backtracking(
                [z, y], tauf, H, proxf, 0, beta=beta, niterback=niterback
            )
        # update x parameter in H function
        H.updatex(x.copy())

        # y step
        z = y + a[1] * (y - yold)
        if gammag is not None:
            dk = gammag * H.lx(x)
            yold = y.copy()
            y = z - (1.0 / dk) * H.grady(z)
            if proxg is not None:
                y = proxg.prox(y, 1.0 / dk)
        else:
            yold = y.copy()
            y, taug = _backtracking(
                [x, y], taug, H, proxf, 1, beta=beta, niterback=niterback
            )
        # update y parameter in H function
        H.updatey(y.copy())

        # run callback
        if callback is not None:
            callback(x, y)

        if show:
            pf = proxf(x) if proxf is not None else 0.0
            pg = proxg(y) if proxg is not None else 0.0
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = "%6g  %5.5e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e" % (
                    iiter + 1,
                    x[0],
                    y[0],
                    pf,
                    pg,
                    H(x, y),
                    ck,
                    dk,
                )
                print(msg)
    if show:
        print("\nTotal time (s) = %.2f" % (time.time() - tstart))
        print("---------------------------------------------------------\n")
    return x, y
