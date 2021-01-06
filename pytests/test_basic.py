import pytest

import numpy as np
from numpy.testing import assert_array_equal
from pylops import MatrixMult

from pyproximal.proximal import Euclidean, Quadratic, L2


par1 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'ny': 21, 'nx': 11, 'nt': 20, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1)])
def test_negativetau(par):
    """Check error is raised when passing negative tau
    """
    l1 = Euclidean(sigma=1.)
    x = np.arange(-5, 5, 0.1)

    with pytest.raises(ValueError):
        l1.prox(x, -1)


@pytest.mark.parametrize("par", [(par1)])
def test_add(par):
    """Check __add__ operator against one proximal operator to which we
    can add a dot-product (e.g., Quadratic)
    """
    A = np.random.normal(0, 1, (par['nx'], par['nx']))
    A = A.T @ A
    quad = Quadratic(Op=MatrixMult(A), b=np.ones(par['nx']), niter=500)
    quad1 = Quadratic(Op=MatrixMult(A), niter=500)
    quad1 = quad1 + np.ones(par['nx'])

    # prox / dualprox
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    tau = 2.
    assert_array_equal(quad.prox(x, tau), quad1.prox(x, tau))
    assert_array_equal(quad.proxdual(x, tau), quad1.proxdual(x, tau))


@pytest.mark.parametrize("par", [(par1)])
def test_mul(par):
    """Check __mul__ operator against one proximal operator to which we
    can multiply a scalar (e.g., L2)
    """
    sigma = 2.
    l2 = L2(sigma=sigma)
    l21 = sigma * L2()

    # prox / dualprox
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    tau = 2.
    assert_array_equal(l2.prox(x, tau), l21.prox(x, tau))
    assert_array_equal(l2.proxdual(x, tau), l21.proxdual(x, tau))
