import time
import numpy as np


def _backtracking(x, tau, proxf, proxg, epsg, beta=0.5, niterback=10):
    r"""Backtracking

    Line-search algorithm for finding step sizes in proximal algorithms when
    the Lipschitz constant of the operator is unknown (or expensive to
    estimate).

    """
    def ftilde(x, y, f, tau):
        xy = x - y
        return f(y) + np.dot(f.grad(y), xy) + \
               (1. / (2. * tau)) * np.linalg.norm(xy) ** 2

    iiterback = 0
    while iiterback < niterback:
        z = proxg.prox(x - tau * proxf.grad(x), epsg * tau)
        ft = ftilde(z, x, proxf, tau)
        if proxf(z) <= ft:
            break
        tau *= beta
        iiterback += 1
    return z, tau


def ProximalPoint(prox, x0, tau, niter=10, show=False):
    r"""Proximal point algorithm

    Solves the following minimization problem using Proximal point algorithm:

    .. math::

        \mathbf{x} = arg min_\mathbf{x} f(\mathbf{x})

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
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Proximal point algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = prox_{\tau f}(\mathbf{x}^k)

    """
    if show:
        tstart = time.time()
        print('Proximal point algorithm\n'
              '---------------------------------------------------------\n'
              'Proximal operator: %s\n'
              'tau = %10e\tniter = %d\n' % (type(prox), tau, niter))
        head = '   Itn       x[0]        f'
        print(head)
    x = x0.copy()
    for iiter in range(niter):
        x = prox.prox(x, tau)
        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = '%6g  %12.5e  %10.3e' % \
                      (iiter + 1, x[0], prox(x))
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def ProximalGradient(proxf, proxg, x0, tau=None, beta=0.5,
                     epsg=1., niter=10, niterback=100,
                     callback=None, show=False):
    r"""Proximal gradient

    Solves the following minimization problem using Proximal gradient
    algorithm:

    .. math::

        \mathbf{x} = arg min_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

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
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration.
    beta : obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    epsg : :obj:`float`, optional
        Scaling factor of g function
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model

    Notes
    -----
    The Proximal point algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = prox_{\tau^k g}(\mathbf{x}^k -
        \tau^k \nabla f(\mathbf{x}^k))

    where at each iteration :math:`\tau^k` can be estimated by back-tracking
    as follows:

    .. math::

        \begin{aligned}
        &\tau = \tau^{k-1} &\\
        &repeat \; \mathbf{z} = prox_{\tau g}(\mathbf{x}^k -
        \tau^k \nabla f(\mathbf{x}^k)), \tau = \beta \tau \quad if \;
        f(\mathbf{z}) \leq \tilde{f}_\tau(\mathbf{z}, \mathbf{x}^k) \\
        &\tau^k = \tau, \quad \mathbf{x}^{k+1} = \mathbf{z} &\\
        \end{aligned}

    where :math:`\tilde{f}_\tau(\mathbf{x}, \mathbf{y}) = f(\mathbf{y}) +
    \nabla f(\mathbf{y})^T (\mathbf{x} - \mathbf{y}) +
    1/(2\tau)||\mathbf{x} - \mathbf{y}||_2^2`.

    """
    if show:
        tstart = time.time()
        print('Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tbeta=%10e\n'
              'epsg = %10e\tniter = %d\t'
              'niterback = %d\n' % (type(proxf), type(proxg),
                                    0 if tau is None else tau, beta, epsg,
                                    niter, niterback))
        head = '   Itn       x[0]          f           g       J=f+eps*g'
        print(head)

    backtracking = False
    if tau is None:
        backtracking = True
        tau = 1.

    x = x0.copy()
    for iiter  in range(niter):
        if not backtracking:
            x = proxg.prox(x - tau * proxf.grad(x), epsg * tau)
        else:
            x, tau = _backtracking(x, tau, proxf, proxg, epsg,
                                   beta=beta, niterback=niterback)

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0], pf, pg, pf + epsg * pg)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def AcceleratedProximalGradient(proxf, proxg, x0, tau=None, beta=0.5,
                                epsg=1., niter=10, niterback=100,
                                callback=None, show=False):
    r"""Accelerated Proximal gradient

    Solves the following minimization problem using Accelerated Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = arg min_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`f(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function (must have ``grad`` implemented)
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration.
    beta : obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    epsg : :obj:`float`, optional
        Scaling factor of g function
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

    Notes
    -----
    The Accelerated Proximal point algorithm can be expressed by the
    following recursion:

    .. math::

        \mathbf{y}^{k+1} = \mathbf{x}^k + \omega^k
        (\mathbf{x}^k - \mathbf{x}^{k-1})  \\
        \mathbf{x}^{k+1} = prox_{\tau^k f}(\mathbf{y}^{k+1}  -
        \tau^k \nabla f(\mathbf{y}^{k+1}))

    where :math:`\omega^k = k / (k + 3)`.

    """
    if show:
        tstart = time.time()
        print('Accelerated Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tepsg = %10e\tniter = %d\n' % (type(proxf),
                                                         type(proxg),
                                                         0 if tau is None else tau,
                                                         epsg, niter))
        head = '   Itn       x[0]          f           g       J=f+eps*g'
        print(head)

    backtracking = False
    if tau is None:
        backtracking = True
        tau = 1.

    x = x0.copy()
    xold = np.zeros_like(x)
    for iiter in range(niter):
        omega = iiter / (iiter + 3)
        y = x + omega * (x - xold)
        xold = x.copy()
        if not backtracking:
            x = proxg.prox(y - tau * proxf.grad(y), epsg * tau)
        else:
            x, tau = _backtracking(y, tau, proxf, proxg, epsg,
                                   beta=beta, niterback=niterback)
        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0], pf, pg, pf + epsg * pg)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def ADMM(proxf, proxg, x0, tau, niter=10, callback=None, show=False):
    r"""Alternating Direction Method of Multipliers

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers (also known as Douglas-Rachford splitting):

    .. math::

        \mathbf{x} = arg min_\mathbf{x} f(\mathbf{x}) + g(\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    function that has a known proximal operator.

    Parameters
    ----------
    proxf : :obj:`pyproximal.ProxOperator`
        Proximal operator of f function
    proxg : :obj:`pyproximal.ProxOperator`
        Proximal operator of g function
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
    The ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
        \mathbf{z}^{k+1} = prox_{\tau g}(\mathbf{x}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

    Note that ``x`` and ``z`` converge to each other, but if iterations are
    stopped too early ``x`` is guaranteed to belong to the domain of ``f``
    while ``z`` is guaranteed to belong to the domain of ``g``. Depending on
    the problem either of the two may be the best solution.

    """
    if show:
        tstart = time.time()
        print('ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tniter = %d\n' % (type(proxf), type(proxg),
                                            tau, niter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    x = x0.copy()
    u = z = np.zeros_like(x)
    for iiter in range(niter):
        x = proxf.prox(z - u, tau)
        z = proxg.prox(x + u, tau)
        u = u + x - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0], pf, pg, pf + pg)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z


def LinearizedADMM(proxf, proxg, A, x0, tau, mu, niter=10,
                   callback=None, show=False):
    r"""Linearized Alternating Direction Method of Multipliers

    Solves the following minimization problem using Linearized Alternating
    Direction Method of Multipliers (also known as Douglas-Rachford splitting):

    .. math::

        \mathbf{x} = arg min_\mathbf{x} f(\mathbf{x}) + g(\mathbf{A}\mathbf{x})

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
    The Linearized-ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = prox_{\mu f}(\mathbf{x}^{k} - \frac{\mu}{\tau}
        \mathbf{A}^T(\mathbf{A} \mathbf{x}^k - \mathbf{z}^k + \mathbf{u}^k))\\
        \mathbf{z}^{k+1} = prox_{\tau g}(\mathbf{A} \mathbf{x}^{k+1} +
        \mathbf{u}^k)\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{A}\mathbf{x}^{k+1} -
        \mathbf{z}^{k+1}

    """
    if show:
        tstart = time.time()
        print('Linearized-ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'tau = %10e\tmu = %10e\tniter = %d\n' % (type(proxf),
                                                       type(proxg),
                                                       type(A),
                                                       tau, mu, niter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)
    x = x0.copy()
    Ax = A.matvec(x)
    u = z = np.zeros_like(Ax)
    for iiter in range(niter):
        x = proxf.prox(x - mu / tau * A.rmatvec(Ax - z + u), mu)
        Ax = A.matvec(x)
        z = proxg.prox(Ax + u, tau)
        u = u + Ax - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(Ax)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0], pf, pg, pf + pg)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, z