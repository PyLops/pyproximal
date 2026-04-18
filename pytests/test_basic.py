import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pylops import MatrixMult

from pyproximal import ProxOperator
from pyproximal.proximal import L1, L2, Euclidean, Quadratic

par1 = {"ny": 21, "nx": 11, "nt": 20, "imag": 0, "dtype": "float32"}  # real
par2 = {"ny": 21, "nx": 11, "nt": 20, "imag": 1j, "dtype": "complex64"}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [par1])
def test_ProxOperator(par):
    """Check errors are raised for non-implemented methods"""
    pop = ProxOperator()

    with pytest.raises(
        NotImplementedError, match="This ProxOperator's __call__ method"
    ):
        pop(np.ones(4))


@pytest.mark.parametrize("par", [par1])
def test_negativetau(par):
    """Check error is raised when passing negative tau"""
    l1 = Euclidean(sigma=1.0)
    x = np.arange(-5, 5, 0.1)

    with pytest.raises(ValueError, match="tau must be positive"):
        l1.prox(x, -1)


@pytest.mark.parametrize("par", [par1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_add_sub(par, sign):
    """Check __add__ and  __sub__ operators against one proximal
    operator to which we can add a dot-product (e.g., Quadratic)
    """
    np.random.seed(10)

    A = np.random.normal(0, 1, (par["nx"], par["nx"]))
    A = A.T @ A
    quad = Quadratic(Op=MatrixMult(A), b=sign * np.ones(par["nx"]), niter=500)
    quad1 = Quadratic(Op=MatrixMult(A), niter=500)
    if sign == 1:
        quad1 = quad1 + np.ones(par["nx"])
    else:
        quad1 = quad1 - np.ones(par["nx"])

    # prox / dualprox
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    tau = 2.0
    assert_array_equal(quad.prox(x, tau), quad1.prox(x, tau))
    assert_array_equal(quad.proxdual(x, tau), quad1.proxdual(x, tau))


@pytest.mark.parametrize("par", [par1])
def test_mul(par):
    """Check __mul__ operator against one proximal operator to which we
    can multiply a scalar (e.g., L2)
    """
    np.random.seed(10)

    sigma = 2.0
    l2 = L2(sigma=sigma)
    l21 = sigma * L2()

    # prox / dualprox
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    tau = 2.0
    assert_array_equal(l2.prox(x, tau), l21.prox(x, tau))
    assert_array_equal(l2.proxdual(x, tau), l21.proxdual(x, tau))


@pytest.mark.parametrize("par", [par1])
def test_adjoint(par):
    """Check _adjoint operator against one proximal operator
    that implements proxdual
    """
    np.random.seed(10)

    sigma = 2.0
    l1 = L1(sigma=sigma)
    l1H = l1.H

    # prox / dualprox
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    tau = 2.0
    assert_array_equal(l1.proxdual(x, tau), l1H.prox(x, tau))
    assert_array_equal(l1.prox(x, tau), l1H.proxdual(x, tau))


@pytest.mark.parametrize("par", [par1])
def test_precomposition(par):
    """Check precomposition method and that it raises an error
    when type of a and b is incorrect
    """
    np.random.seed(10)

    l1 = L1()
    a, b = 1.0, 5.0
    l1prec = l1.precomposition(a=a, b=b)
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])

    # norm
    assert l1prec(x) == l1prec.f(a * x + b)

    # grad
    assert_array_equal(l1prec.grad(x), a * l1.grad(a * x + b))

    # check errors
    with pytest.raises(NotImplementedError, match="a must be of type float and b"):
        _ = l1.precomposition(a=1, b=np.ones(5))  # should be float

        _ = l1.precomposition(
            a=1.0,
            b=[1, 2, 3],  # should be float
        )  # should be float, np.ndarray or cp.ndarray


@pytest.mark.parametrize("par", [par1])
def test_postcomposition(par):
    """Check postcomposition method and that it raises an error
    when type of sigma is incorrect
    """
    np.random.seed(10)

    l1 = L1()
    sigma = 2.0
    l1post = l1.postcomposition(sigma=sigma)
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])

    # norm
    assert l1post(x) == sigma * l1(x)

    # grad
    assert_array_equal(l1post.grad(x), sigma * l1.grad(x))

    # check errors
    with pytest.raises(NotImplementedError, match="sigma must be of type float"):
        _ = l1.postcomposition(
            sigma=1,  # should be float
        )
        _ = l1.postcomposition(
            sigma=np.ones(5),  # should be float
        )


@pytest.mark.parametrize("par", [par1])
def test_affine_addition_type(par):
    """Check affine_addition method and that it raises an error
    when type of v is incorrect
    """
    l1 = L1()

    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    v = np.ones(par["nx"]).astype(par["dtype"])
    l1aff = l1 + v

    # norm
    assert l1aff(x) == l1(x) + np.dot(v, x)

    # grad
    assert_array_equal(l1aff.grad(x), l1.grad(x) + v)

    # error
    with pytest.raises(NotImplementedError, match="v must be a numpy.ndarray"):
        _ = l1.affine_addition(
            v=1,  # should be np.ndarray or cp.ndarray
        )
