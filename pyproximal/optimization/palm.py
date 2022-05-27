import time


def PALM(H, proxf, proxg, x0, y0, gammaf=1., gammag=1.,
         niter=10, callback=None, show=False):
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
        Positive scalar weight for ``f`` function update
    gammag : :obj:`float`, optional
        Positive scalar weight for ``g`` function update
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme
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
    and :math:`\nabla_y H`, respectively.

    .. [1] Bolte, J., Sabach, S., and Teboulle, M. "Proximal alternating
       linearized minimization for nonconvex and nonsmooth problems",
       Mathematical Programming, vol. 146, pp. 459–494. 2014.

    """
    if show:
        tstart = time.time()
        print('PALM algorithm\n'
              '---------------------------------------------------------\n'
              'Bilinear operator: %s\n'
              'Proximal operator (f): %s\n'
              'Proximal operator (g): %s\n'
              'gammaf = %10e\tgammaf = %10e\tniter = %d\n' %
              (type(H), type(proxf), type(proxg), gammaf, gammag, niter))
        head = '   Itn      x[0]       y[0]        f         g         H         ck         dk'
        print(head)

    x, y = x0.copy(), y0.copy()
    for iiter in range(niter):
        ck = gammaf * H.ly(y)
        x = x - (1 / ck) * H.gradx(x.ravel())
        if proxf is not None:
            x = proxf.prox(x, ck)
        H.updatex(x.copy())
        dk = gammag * H.lx(x)
        y = y - (1 / dk) * H.grady(y.ravel())
        if proxg is not None:
            y = proxg.prox(y, dk)
        H.updatey(y.copy())

        # run callback
        if callback is not None:
            callback(x, y)

        if show:
            pf = proxf(x) if proxf is not None else 0.
            pg = proxg(y) if proxg is not None else 0.
            if iiter < 10 or niter - iiter < 10 or iiter % (niter // 10) == 0:
                msg = '%6g  %5.5e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e  %5.2e' % \
                      (iiter + 1, x[0], y[0], pf if pf is not None else 0.,
                       pg if pg is not None else 0., H(x, y), ck, dk)
                print(msg)
    if show:
        print('\nTotal time (s) = %.2f' % (time.time() - tstart))
        print('---------------------------------------------------------\n')
    return x, y
