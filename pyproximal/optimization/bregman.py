import time
import numpy as np
from copy import deepcopy


def Bregman(proxf, proxg, x0, solver, A=None, alpha=1., niterouter=10,
            warm=False, tolx=1e-10, tolf=1e-10, bregcallback=None, show=False,
            **kwargs_solver):
    r"""Bregman iterations with Proximal Solver

    Solves one of the following minimization problem using Bregman iterations
    and a Proximal solver of choice for the inner iterations:

    .. math::

        1. \; \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \alpha g(\mathbf{x})

    or

    .. math::

        2. \;     \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) +
        \alpha g(\mathbf{A}\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    function that has a known proximal operator and :math:`\mathbf{A}` is a
    linear operator. The function :math:`g(\mathbf{y})` is converted into
    its equivalent Bregman distance :math:`D_g^{\mathbf{q}^{k}}(\mathbf{y},
    \mathbf{y}_k) = g(\mathbf{y}) - g(\mathbf{y}^k) - (\mathbf{q}^{k})^T
    (\mathbf{y} - \mathbf{y}_k)`.

    If :math:`f(x)` has a uniquely defined gradient the
    :func:`pyproximal.optimization.primal.ProximalGradient` and
    :func:`pyproximal.optimization.primal.AcceleratedProximalGradient` solvers
    can be used to solve the
    first problem, otherwise the :func:`pyproximal.optimization.primal.ADMM`
    is required. On the other hand, only the
    :func:`pyproximal.optimization.primal.LinearizedADMM` solver can be used
    for the second problem.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    solver : :func:`pyprox.optimization.primal`
        Solver used to solve the inner loop optimization problems
    A : :obj:`pylops.LinearOperator`, optional
        Linear operator of g
    alpha : :obj:`float`, optional
        Scalar of g function
    niterouter : :obj:`int`, optional
        Number of iterations of outerloop
    warm : :obj:`bool`, optional
        Warm start  - i.e., previous estimate is used
        as starting guess of the current optimization (``True``) or not - i.e.,
        provided starting guess is used as starting guess of every optimization
        (``False``).
    tolx : :obj:`callable`, optional
        Tolerance on solution update, stop when
        :math:`||\mathbf{x}^{k+1} - \mathbf{x}^k||_2<tol_x` is satisfied
    tolf : :obj:`callable`, optional
        Tolerance on ``f`` function, stop when
        :math:`f(\mathbf{x}^{k+1})_2<tol_f` is satisfied
    bregcallback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each Bregman
        iteration where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    **kwargs_solver : :obj:`dict`, optional
        Arbitrary keyword arguments for chosen solver

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Bregman iterations can be expressed with the following recursion

    .. math::

        \mathbf{x}^{k+1} = \argmin_{\mathbf{x}} \quad f + \alpha g -
        \alpha (\mathbf{q}^{k})^T \mathbf{x}\\
        \mathbf{q}^{k+1} = \mathbf{q}^{k} - \frac{1}{\alpha}
        \nabla f(\mathbf{x}^{k+1})

    where the minimization problem can be solved using one the proximal solvers
    in the :mod:`pyproximal.optimization.primal` module.

    """
    if show:
        tstart = time.time()
        print('Bregman\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'Inner Solver: %s\n'
              'alpha = %10e\ttolf = %10e\ttolx = %10e\n'
              'niter = %d\n' % (type(proxf), type(proxg), type(A), solver,
                                alpha, tolf, tolx, niterouter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    # multiply alpha to proxg
    proxg = alpha * proxg

    x = np.copy(x0)
    q = np.zeros_like(x0)
    for iiter in range(niterouter):
        xold = x.copy()
        # solve optimization
        if iiter == 0:
            proxf_q = proxf
        else:
            proxf_q = deepcopy(proxf) - alpha * q.copy()

        if A is None:
            x = solver(proxf_q, proxg, x0=x if warm else x0, **kwargs_solver)
        else:
            x = solver(proxf_q, proxg, A=A, x0=x if warm else x0, **kwargs_solver)
        if isinstance(x, tuple):
            x = x[0]

        # update q
        q = q - (1. / alpha) * proxf.grad(x)

        # run callback
        if bregcallback is not None:
            bregcallback(x)

        pf = proxf(x)
        if show:
            if iiter < 10 or niterouter - iiter < 10 or iiter % 10 == 0:
                pg = proxg(A.matvec(x)) if A is not None else proxg(x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0], pf, pg, pf + pg)
                print(msg)

        if np.linalg.norm(x - xold) < tolx or pf < tolf:
            break

    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x
