import numpy as np

from pylops.utils.backend import get_module, to_numpy


def gradtest_proximal(Op, n, x=None, dtype="float64",
                      delta=1e-6, rtol=1e-6, atol=1e-21,
                      complexflag=False, raiseerror=True,
                      verb=False, backend="numpy"):
    r"""Gradient test for Proximal operator.

    Compute the gradient of ``Op`` using both the provided method and a
    numerical approximation with a perturbation ``delta`` applied to a
    single, randomly selected parameter of the input vector.

    Parameters
    ----------
    Op : :obj:`pyproximal.Proximal`
        Proximal operator to test.
    n : :obj:`int`
        Size of input vector
    x : :obj:`numpy.ndarray`, optional
        Input vector (if ``None``, randomly drawn from a
        Normal distribution)
    dtype : :obj:`str`, optional
        Dtype of vector ``x`` to generate (only used when ``x=None``)
    delta : :obj:`float`, optional
        Perturbation
    rtol : :obj:`float`, optional
        Relative gradtest tolerance
    atol : :obj:`float`, optional
        Absolute gradtest tolerance
    complexflag : :obj:`bool`, optional
        Generate random vectors with real (``False``) or
        complex (``True``) entries
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    passed : :obj:`bool`
        Passed flag.

    Raises
    ------
    AssertionError
        If grad-test is not verified within chosen tolerances.

    Notes
    -----
    A gradient-test is mathematical tool used in the development of numerical
    nonliner operators.

    More specifically, a correct implementation of the gradient for
    a nonlinear operator should verify the following *equality*
    within a numerical tolerance:

    .. math::
        \frac{\partial Op(\mathbf{x})}{\partial \mathbf{x}} =
        \frac{Op(\mathbf{x}+\delta \mathbf{x})-Op(\mathbf{x})}{\delta \mathbf{x}}

    """
    ncp = get_module(backend)

    # get random vectors for x and y
    if x is None:
        x = np.random.normal(0., 1., n).astype(dtype)

        if complexflag:
            x = x + 1j * np.random.normal(0., 1., n).astype(dtype)

    # compute function
    f = Op(x)

    # compute gradient
    g = Op.grad(x)

    # choose location of perturbation, whether to act on x or y and on real or imag part
    iqx = np.random.randint(0, n)
    r_or_i = np.random.randint(0, 2)

    if r_or_i == 0:
        delta1 = delta
    else:
        delta1 = delta * 1j

    # extract gradient value to test
    x[iqx] = x[iqx] + delta1
    grad = g[iqx]

    # compute new function at perturbed location
    fdelta = Op(x)

    # evaluate if gradient test passed
    grad_delta = (fdelta - f) / np.abs(delta)
    grad_diff = grad_delta - (grad.real if r_or_i == 0 else grad.imag)
    passed = np.isclose(grad_diff, 0, rtol, atol)

    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status = "passed" if passed else "failed"
        msg = f"Grad test {passed_status}, Analytic={grad.real if r_or_i == 0 else grad.imag} - " \
              f"Numeric={grad_delta}"
        if not passed and raiseerror:
            raise AssertionError(msg)
        else:
            print(msg)

    return passed


def gradtest_bilinear(Op, delta=1e-6, rtol=1e-6, atol=1e-21,
                      complexflag=False, raiseerror=True,
                      verb=False, backend="numpy"):
    r"""Gradient test for Bilinear operator.

    Compute the gradient of ``Op`` using both the provided method and a
    numerical approximation with a perturbation ``delta`` applied to a
    single, randomly selected parameter of either the ``x`` or ``y``
    vectors.

    Parameters
    ----------
    Op : :obj:`pyproximal.utils.BilinearOperator`
        Bilinear operator to test.
    delta : :obj:`float`, optional
        Perturbation
    rtol : :obj:`float`, optional
        Relative gradtest tolerance
    atol : :obj:`float`, optional
        Absolute gradtest tolerance
    complexflag : :obj:`bool`, optional
        Generate random vectors with real (``False``) or
        complex (``True``) entries
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity
    backend : :obj:`str`, optional
        Backend used for dot test computations (``numpy`` or ``cupy``). This
        parameter will be used to choose how to create the random vectors.

    Returns
    -------
    passed : :obj:`bool`
        Passed flag.

    Raises
    ------
    AssertionError
        If grad-test is not verified within chosen tolerances.

    Notes
    -----
    A gradient-test is mathematical tool used in the development of numerical
    bilinear operators.

    More specifically, a correct implementation of the gradient for
    a bilinear operator should verify the following *equalities*
    within a numerical tolerance:

    .. math::
        \frac{\partial Op(\mathbf{x})}{\partial \mathbf{x}} =
        \frac{Op(\mathbf{x}+\delta \mathbf{x}, \mathbf{y})-
        Op(\mathbf{x})}{\delta \mathbf{x}, \mathbf{y}}

    and

    .. math::
        \frac{\partial Op(\mathbf{x}, \mathbf{y})}{\partial \mathbf{y}} =
        \frac{Op(\mathbf{x}, \mathbf{y}+\delta \mathbf{y})-
        Op(\mathbf{x}, \mathbf{y})}{\delta \mathbf{y}}

    """
    ncp = get_module(backend)

    nx = Op.sizex
    ny = Op.sizey

    # extract x and y from Op
    x, y = Op.x.ravel(), Op.y.ravel()

    # compute function at x and y
    f = Op(x, y)

    # compute gradients at x and y
    gx = Op.gradx(x)
    gy = Op.grady(y)

    # choose location of perturbation, whether to act on x or y and on real or imag part
    iqx, iqy = np.random.randint(0, nx), np.random.randint(0, ny)
    x_or_y = np.random.randint(0, 2)

    delta1 = delta
    if complexflag:
        r_or_i = np.random.randint(0, 2)
        if r_or_i == 1:
            delta1 = delta * 1j

    # extract gradient value to test
    if x_or_y == 0:
        x[iqx] = x[iqx] + delta1
        grad = gx[iqx]
    else:
        y[iqy] = y[iqy] + delta1
        grad = gy[iqy]

    # compute new function at perturbed location
    fdelta = Op(x, y)

    # evaluate if gradient test passed
    grad_delta = (fdelta - f) / np.abs(delta)
    grad_diff = grad_delta - (grad.real if not complexflag or r_or_i == 0 else grad.imag)
    passed = np.isclose(grad_diff, 0, rtol, atol)

    # verbosity or error raising
    if (not passed and raiseerror) or verb:
        passed_status = "passed" if passed else "failed"
        msg = f"Grad test {passed_status}, Analytic={grad.real if r_or_i == 0 else grad.imag} - " \
              f"Numeric={grad_delta}"
        if not passed and raiseerror:
            raise AssertionError(msg)
        else:
            print(msg)

    return passed
