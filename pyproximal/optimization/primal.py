import time
import warnings
import numpy as np

from math import sqrt
from pylops.optimization.leastsquares import RegularizedInversion
from pyproximal.proximal import L2


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


def ProximalPoint(prox, x0, tau, niter=10, callback=None, show=False):
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

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^k)

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

        # run callback
        if callback is not None:
            callback(x)

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
                     acceleration=None,
                     callback=None, show=False):
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
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration. Finally note that :math:`\tau` can be chosen to be a vector
        when dealing with problems with multiple right-hand-sides
    beta : :obj:`float`, optional
        Backtracking parameter (must be between 0 and 1)
    epsg : :obj:`float` or :obj:`np.ndarray`, optional
        Scaling factor of g function
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    niterback : :obj:`int`, optional
        Max number of iterations of backtracking
    acceleration : :obj:`str`, optional
        Acceleration (``None``, ``vandenberghe`` or ``fista``)
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


        \mathbf{x}^{k+1} = \prox_{\tau^k \epsilon g}(\mathbf{y}^{k+1}  -
        \tau^k \nabla f(\mathbf{y}^{k+1})) \\
        \mathbf{y}^{k+1} = \mathbf{x}^k + \omega^k
        (\mathbf{x}^k - \mathbf{x}^{k-1})

    where at each iteration :math:`\tau^k` can be estimated by back-tracking
    as follows:

    .. math::

        \begin{aligned}
        &\tau = \tau^{k-1} &\\
        &repeat \; \mathbf{z} = \prox_{\tau \epsilon g}(\mathbf{x}^k -
        \tau \nabla f(\mathbf{x}^k)), \tau = \beta \tau \quad if \;
        f(\mathbf{z}) \leq \tilde{f}_\tau(\mathbf{z}, \mathbf{x}^k) \\
        &\tau^k = \tau, \quad \mathbf{x}^{k+1} = \mathbf{z} &\\
        \end{aligned}

    where :math:`\tilde{f}_\tau(\mathbf{x}, \mathbf{y}) = f(\mathbf{y}) +
    \nabla f(\mathbf{y})^T (\mathbf{x} - \mathbf{y}) +
    1/(2\tau)||\mathbf{x} - \mathbf{y}||_2^2`.

    Different accelerations are provided:

    - ``acceleration=None``: :math:`\omega^k = 0`;
    - `acceleration=vandenberghe`` [1]_: :math:`\omega^k = k / (k + 3)` for `
    - ``acceleration=fista``: :math:`\omega^k = (t_{k-1}-1)/t_k` for  where
      :math:`t_k = (1 + \sqrt{1+4t_{k-1}^{2}}) / 2` [2]_

    .. [1] Vandenberghe, L., "Fast proximal gradient methods", 2010.
    .. [2] Beck, A., and Teboulle, M. "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems", SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """
    # check if epgs is a ve
    if np.asarray(epsg).size == 1.:
        epsg_print = str(epsg)
    else:
        epsg_print = 'Multi'

    if acceleration not in [None, 'None', 'vandenberghe', 'fista']:
        raise NotImplementedError('Acceleration should be None, vandenberghe '
                                  'or fista')
    if show:
        tstart = time.time()
        print('Accelerated Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %s\tbeta=%10e\n'
              'epsg = %s\tniter = %d\t'
              'niterback = %d\n' % (type(proxf), type(proxg),
                                    'Adaptive' if tau is None else str(tau), beta,
                                    epsg_print, niter, niterback))
        head = '   Itn       x[0]          f           g       J=f+eps*g'
        print(head)

    backtracking = False
    if tau is None:
        backtracking = True
        tau = 1.

    # initialize model
    t = 1.
    x = x0.copy()
    y = x.copy()

    # iterate
    for iiter in range(niter):
        xold = x.copy()

        # proximal step
        if not backtracking:
            x = proxg.prox(y - tau * proxf.grad(y), epsg * tau)
        else:
            x, tau = _backtracking(y, tau, proxf, proxg, epsg,
                                   beta=beta, niterback=niterback)
        
        # update y
        if acceleration == 'vandenberghe':
            omega = iiter / (iiter + 3)
        elif acceleration == 'fista':
            told = t
            t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
            omega = ((told - 1.) / t)
        else:
            omega = 0
        y = x + omega * (x - xold)

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = proxf(x), proxg(x)
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0] if x.ndim == 1 else x[0, 0],
                       pf, pg[0] if epsg_print == 'Multi' else pg,
                       pf + np.sum(epsg * pg))
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def AcceleratedProximalGradient(proxf, proxg, x0, tau=None, beta=0.5,
                                epsg=1., niter=10, niterback=100,
                                acceleration='vandenberghe',
                                callback=None, show=False):
    r"""Accelerated Proximal gradient

    This is a thin wrapper around :func:`pyproximal.optimization.primal.ProximalGradient` with
    ``vandenberghe`` or ``fista``acceleration. See :func:`pyproximal.optimization.primal.ProximalGradient`
    for details.

    """
    warnings.warn('AcceleratedProximalGradient has been integrated directly into ProximalGradient '
                  'from v0.5.0. It is recommended to start using ProximalGradient by selecting the '
                  'appropriate acceleration parameter as this behaviour will become default in '
                  'version v1.0.0 and AcceleratedProximalGradient will be removed.', FutureWarning)
    return ProximalGradient(proxf, proxg, x0, tau=tau, beta=beta,
                            epsg=epsg, niter=niter, niterback=niterback,
                            acceleration=acceleration,
                            callback=callback, show=show)


def GeneralizedProximalGradient(proxfs, proxgs, x0, tau=None,
                                epsg=1., niter=10,
                                acceleration=None,
                                callback=None, show=False):
    r"""Generalized Proximal gradient

    Solves the following minimization problem using Generalized Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^n f_i(\mathbf{x}) 
        + \sum_{j=1}^m \tau_j g_j(\mathbf{x}),~~n,m \in \mathbb{N}^+

    where the :math:`f_i(\mathbf{x})` are smooth convex functions with a uniquely
    defined gradient and the :math:`g_j(\mathbf{x})` are any convex function that
    have a known proximal operator.

    Parameters
    ----------
    proxfs : :obj:`List of pyproximal.ProxOperator`
        Proximal operators of the :math:`f_i` functions (must have ``grad`` implemented)
    proxgs : :obj:`List of pyproximal.ProxOperator`
        Proximal operators of the :math:`g_j` functions
    x0 : :obj:`numpy.ndarray`
        Initial vector
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\sum_{i=1}^n \nabla f_i`. When ``tau=None``,
        backtracking is used to adaptively estimate the best tau at each
        iteration.
    epsg : :obj:`float` or :obj:`np.ndarray`, optional
        Scaling factor of g function
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    acceleration:  :obj:`str`, optional
        Acceleration (``vandenberghe`` or ``fista``)
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
    The Generalized Proximal point algorithm can be expressed by the
    following recursion:

    .. math::
        \text{for } j=1,\cdots,n, \\
        ~~~~\mathbf z_j^{k+1} = \mathbf z_j^{k} + \eta_k (prox_{\frac{\tau^k}{\omega_j} g_j}(2 \mathbf{x}^{k} - z_j^{k}) 
        - \tau^k \sum_{i=1}^n \nabla f_i(\mathbf{x}^{k})) - \mathbf{x}^{k} \\
        \mathbf{x}^{k+1} = \sum_{j=1}^n \omega_j f_j \\
        
    where :math:`\sum_{j=1}^n \omega_j=1`. 
    """
    # check if epgs is a vector
    if np.asarray(epsg).size == 1.:
        epsg_print = str(epsg)
    else:
        epsg_print = 'Multi'

    if acceleration not in [None, 'None', 'vandenberghe', 'fista']:
        raise NotImplementedError('Acceleration should be None, vandenberghe '
                                  'or fista')
    if show:
        tstart = time.time()
        print('Generalized Proximal Gradient\n'
              '---------------------------------------------------------\n'
              'Proximal operators (f): %s\n'
              'Proximal operators (g): %s\n'
              'tau = %10e\nepsg = %s\tniter = %d\n' % ([type(proxf) for proxf in proxfs],
                                                         [type(proxg) for proxg in proxgs],
                                                         0 if tau is None else tau,
                                                         epsg_print, niter))
        head = '   Itn       x[0]          f           g       J=f+eps*g'
        print(head)

    if tau is None:
        tau = 1.

    # initialize model
    t = 1.
    x = x0.copy()
    y = x.copy()
    zs = [x.copy() for _ in range(len(proxgs))]

    # iterate
    for iiter in range(niter):
        xold = x.copy()

        # proximal step
        grad = np.zeros_like(x)
        for i, proxf in enumerate(proxfs):
            grad += proxf.grad(x)

        sol = np.zeros_like(x)
        for i, proxg in enumerate(proxgs):
            tmp = 2 * y - zs[i] - tau * grad
            tmp[:] = proxg.prox(tmp, tau *len(proxgs) )
            zs[i] += epsg * (tmp - y)
            sol += zs[i] / len(proxgs)
        x[:] = sol.copy()

        # update y
        if acceleration == 'vandenberghe':
            omega = iiter / (iiter + 3)
        elif acceleration== 'fista':
            told = t
            t = (1. + np.sqrt(1. + 4. * t ** 2)) / 2.
            omega = ((told - 1.) / t)
        else:
            omega = 0
        
        y = x + omega * (x - xold)

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = np.sum([proxf(x) for proxf in proxfs]), np.sum([proxg(x) for proxg in proxgs])
                msg = '%6g  %12.5e  %10.3e  %10.3e  %10.3e' % \
                      (iiter + 1, x[0] if x.ndim == 1 else x[0, 0],
                       pf, pg[0] if epsg_print == 'Multi' else pg,
                       pf + np.sum(epsg * pg))
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x


def HQS(proxf, proxg, x0, tau, niter=10, z0=None, gfirst=True,
        callback=None, callbackz=False, show=False):
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
        Initial vector
    tau : :obj:`float` or :obj:`numpy.ndarray`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
        the Lipschitz constant of :math:`\nabla f`. Finally note that
        :math:`\tau` can be chosen to be a vector of size ``niter`` such that
        different :math:`\tau` is used at different iterations (i.e., continuation
        strategy)
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    z0 : :obj:`numpy.ndarray`, optional
        Initial z vector (not required when ``gfirst=True``
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
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
    The HQS algorithm can be expressed by the following recursion [1]_:

    .. math::

        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k}) \\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k+1})

    for ``gfirst=False``, or

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k}) \\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1})

    for ``gfirst=False``. Note that ``x`` and ``z`` converge to each other,
    however if iterations are stopped too early ``x`` is guaranteed to belong to
    the domain of ``f`` while ``z`` is guaranteed to belong to the domain of ``g``.
    Depending on the problem either of the two may be the best solution.

    .. [1] D., Geman, and C., Yang, "Nonlinear image recovery with halfquadratic
         regularization", IEEE Transactions on Image Processing,
         4, 7, pp. 932-946, 1995.

    """
    # check if epgs is a ve
    if np.asarray(tau).size == 1.:
        tau_print = str(tau)
        tau = tau * np.ones(niter)
    else:
        tau_print = 'Variable'

    if show:
        tstart = time.time()
        print('HQS\n'
              '---------------------------------------------------------\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'tau = %s\tniter = %d\n' % (type(proxf), type(proxg),
                                          tau_print, niter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    x = x0.copy()
    if z0 is not None:
        z = z0.copy()
    else:
        z = np.zeros_like(x)
    for iiter in range(niter):
        if gfirst:
            z = proxg.prox(x, tau[iiter])
            x = proxf.prox(z, tau[iiter])
        else:
            x = proxf.prox(z, tau[iiter])
            z = proxg.prox(x, tau[iiter])

        # run callback
        if callback is not None:
            if callbackz:
                callback(x, z)
            else:
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


def ADMM(proxf, proxg, x0, tau, niter=10, gfirst=False,
         callback=None, callbackz=False, show=False):
    r"""Alternating Direction Method of Multipliers

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers (also known as Douglas-Rachford splitting):

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    ADMM can also solve the problem of the form above with a more general
    constraint: :math:`\mathbf{Ax}+\mathbf{Bz}=c`. This routine implements
    the special case where :math:`\mathbf{A}=\mathbf{I}`, :math:`\mathbf{B}=-\mathbf{I}`,
    and :math:`c=0`, as a general algorithm can be obtained for any choice of
    :math:`f` and :math:`g` provided they have a known proximal operator.

    On the other hand, for more general choice of :math:`\mathbf{A}`, :math:`\mathbf{B}`,
    and :math:`c`, the iterations are not generalizable, i.e. thye depends on the choice of
    :math:`f` and :math:`g` functions. For this reason, we currently only provide an additional
    solver for the special case where :math:`f` is a :class:`pyproximal.proximal.L2`
    operator with a linear operator  :math:`\mathbf{G}` and data :math:`\mathbf{y}`,
    :math:`\mathbf{B}=-\mathbf{I}` and :math:`c=0`,
    called :func:`pyproximal.optimization.primal.ADMML2`. Note that for the very same choice
    of :math:`\mathbf{B}` and :math:`c`, the :func:`pyproximal.optimization.primal.LinearizedADMM`
    can also be used (and this does not require a specific choice of :math:`f`).

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
    gfirst : :obj:`bool`, optional
        Apply Proximal of operator ``g`` first (``True``) or Proximal of
        operator ``f`` first (``False``)
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    callbackz : :obj:`bool`, optional
        Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
    show : :obj:`bool`, optional
        Display iterations log

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

    Notes
    -----
    The ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

    Note that ``x`` and ``z`` converge to each other, however if iterations are
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
        if gfirst:
            z = proxg.prox(x + u, tau)
            x = proxf.prox(z - u, tau)
        else:
            x = proxf.prox(z - u, tau)
            z = proxg.prox(x + u, tau)
        u = u + x - z

        # run callback
        if callback is not None:
            if callbackz:
                callback(x, z)
            else:
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


def ADMML2(proxg, Op, b, A, x0, tau, niter=10, callback=None, show=False, **kwargs_solver):
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
    tau : :obj:`float`, optional
        Positive scalar weight, which should satisfy the following condition
        to guarantees convergence: :math:`\tau \in (0, 1/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    **kwargs_solver
        Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
        to solve the x-update

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    z : :obj:`numpy.ndarray`
        Inverted second model

    See Also
    --------
    ADMM: ADMM
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    The ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \argmin_{\mathbf{x}} \frac{1}{2}||\mathbf{Op}\mathbf{x}
        - \mathbf{b}||_2^2 + \frac{1}{2\tau} ||\mathbf{Ax} - \mathbf{z}^k + \mathbf{u}^k||_2^2\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{Ax}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{Ax}^{k+1} - \mathbf{z}^{k+1}

    """
    if show:
        tstart = time.time()
        print('ADMM\n'
              '---------------------------------------------------------\n'
              'Proximal operator (g): %s\n'
              'tau = %10e\tniter = %d\n' % (type(proxg), tau, niter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    sqrttau = 1. / sqrt(tau)
    x = x0.copy()
    u = z = np.zeros(A.shape[0], dtype=A.dtype)
    for iiter in range(niter):
        # create augumented system
        x = RegularizedInversion(Op, [A, ], b,
                                 dataregs=[z - u, ], epsRs=[sqrttau, ],
                                 x0=x, **kwargs_solver)
        Ax = A @ x
        z = proxg.prox(Ax + u, tau)
        u = u + Ax - z

        # run callback
        if callback is not None:
            callback(x)

        if show:
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                pf, pg = np.linalg.norm(Op @ x - b), proxg(Ax)
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

    See Also
    --------
    ADMM: ADMM
    ADMML2: ADMM with L2 misfit function

    Notes
    -----
    The Linearized-ADMM algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\mu f}(\mathbf{x}^{k} - \frac{\mu}{\tau}
        \mathbf{A}^H(\mathbf{A} \mathbf{x}^k - \mathbf{z}^k + \mathbf{u}^k))\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{A} \mathbf{x}^{k+1} +
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


def TwIST(proxg, A, b, x0, alpha=None, beta=None, eigs=None, niter=10,
          callback=None, show=False, returncost=False):
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
    callback : :obj:`callable`, optional
        Function with signature (``callback(x)``) to call after each iteration
        where ``x`` is the current model vector
    show : :obj:`bool`, optional
        Display iterations log
    returncost : :obj:`bool`, optional
        Return cost function

    Returns
    -------
    x : :obj:`numpy.ndarray`
        Inverted model
    j : :obj:`numpy.ndarray`
        Cost function

    Notes
    -----
    The TwIST algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = (1-\alpha) \mathbf{x}^{k-1} +
        (\alpha-\beta) \mathbf{x}^k +
        \beta \prox_{g} (\mathbf{x}^k + \mathbf{A}^H
        (\mathbf{b} - \mathbf{A}\mathbf{x}^k)).

    where :math:`\mathbf{x}^{1} = \prox_{g} (\mathbf{x}^0 + \mathbf{A}^T
    (\mathbf{b} - \mathbf{A}\mathbf{x}^0))`.

    The optimal weighting parameters :math:`\alpha` and :math:`\beta` are
    linked to the smallest and largest eigenvalues of
    :math:`\mathbf{A}^H\mathbf{A}` as follows:

    .. math::

        \alpha = 1 + \rho^2 \\
        \beta =\frac{2 \alpha}{\Lambda_{max} + \lambda_{min}}

    where :math:`\rho=\frac{1-\sqrt{k}}{1+\sqrt{k}}` with
    :math:`k=\frac{\lambda_{min}}{\Lambda_{max}}` and
    :math:`\Lambda_{max}=max(1, \lambda_{max})`.

    Experimentally, it has been observed that TwIST is robust to the
    choice of such parameters. Finally, note that in the case of
    :math:`\alpha=1` and :math:`\beta=1`, TwIST is identical to IST.

    """
    # define proxf as L2 proximal
    proxf = L2(Op=A, b=b)

    # find alpha and beta
    if alpha is None or beta is None:
        if eigs is None:
            emin = A.eigs(neigs=1, which='SM')
            emax = max([1, A.eigs(neigs=1, which='LM')])
        else:
            emax, emin = eigs
        k = emin / emax
        rho =  (1 - sqrt(k)) / (1 + sqrt(k))
        alpha = 1 + rho ** 2
        beta = 2 * alpha / (emax + emin)

    # compute proximal of g on initial guess (x_1)
    xold = x0.copy()
    x = proxg.prox(xold - proxf.grad(xold), 1.)

    if show:
        tstart = time.time()
        print('TwIST\n'
              '---------------------------------------------------------\n'
              'Proximal operator (g): %s\n'
              'Linear operator (A): %s\n'
              'alpha = %10e\tbeta = %10e\tniter = %d\n' % (type(proxg),
                                                           type(A),
                                                           alpha, beta, niter))
        head = '   Itn       x[0]          f           g       J = f + g'
        print(head)

    # iterate
    j = None
    if returncost:
        j = np.zeros(niter)
    for iiter in range(niter):
        # compute new x
        xnew = (1 - alpha) * xold + \
               (alpha - beta) * x + \
               beta * proxg.prox(x - proxf.grad(x), 1.)
        # save current x as old (x_i -> x_i-1)
        xold = x.copy()
        # save new x as current (x_i+1 -> x_i)
        x = xnew.copy()

        # compute cost function
        if returncost:
            j[iiter] = proxf(x) + proxg(x)

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
    if returncost:
        return x, j
    else:
        return x
