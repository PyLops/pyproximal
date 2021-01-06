import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import Identity, Diagonal
from pyproximal.utils import moreau
from pyproximal.proximal import Box, Euclidean, L2, L1, L21

par1 = {'nx': 10, 'sigma': 1., 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'sigma': 2., 'dtype': 'float64'}  # odd float64

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Euclidean(par):
    """Euclidean norm and proximal/dual proximal
    """
    eucl = Euclidean(sigma=par['sigma'])

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert eucl(x) == par['sigma'] * np.linalg.norm(x)

    # grad
    assert_array_almost_equal(eucl.grad(x),
                              par['sigma'] * x / np.linalg.norm(x))

    # prox / dualprox
    tau = 2.
    assert moreau(eucl, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2(par):
    """L2 norm and proximal/dual proximal
    """
    l2 = L2(Op=Identity(par['nx'], dtype=par['dtype']),
            b=np.zeros(par['nx'], dtype=par['dtype']),
            sigma=par['sigma'])
    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l2(x) == (par['sigma'] / 2.) * np.linalg.norm(x) ** 2

    # prox
    tau = 2.
    assert_array_almost_equal(l2.prox(x, tau), x / (1. + par['sigma'] * tau))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2_diff(par):
    """L2 norm of difference and proximal/dual proximal
    """
    b = np.ones(par['nx'], dtype=par['dtype'])
    l2 = L2(b=b, sigma=par['sigma'])

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l2(x) == (par['sigma'] / 2.) * np.linalg.norm(x - b) ** 2

    # prox
    tau = 2.
    assert_array_almost_equal(l2.prox(x, tau), (x + par['sigma'] * tau * b) /
                              (1. + par['sigma'] * tau))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2_op(par):
    """L2 norm of Op*x and proximal/dual proximal
    """
    b = np.zeros(par['nx'], dtype=par['dtype'])
    d = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    l2 = L2(Op=Diagonal(d, dtype=par['dtype']),
            b=b, sigma=par['sigma'], niter=500)

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l2(x) == (par['sigma'] / 2.) * np.linalg.norm(d * x) ** 2

    # prox: since Op is a Diagonal operator the denominator becomes
    # 1 + sigma*tau*d[i] for every i
    tau = 2.
    den = 1. + par['sigma'] * tau * d ** 2
    assert_array_almost_equal(l2.prox(x, tau), x / den, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1(par):
    """L1 norm and proximal/dual proximal
    """
    l1 = L1(sigma=par['sigma'])

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l1(x) == par['sigma'] * np.sum(np.abs(x))

    # prox / dualprox
    tau = 2.
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1_diff(par):
    """L1 norm of difference and proximal/dual proximal
    """
    l1 = L1(sigma=par['sigma'],
            g=np.random.normal(0., 1., par['nx']).astype(par['dtype']))

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l1(x) == par['sigma'] * np.sum(np.abs(x))

    # prox / dualprox
    tau = 2.
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L21(par):
    """L21 norm and proximal/dual proximal on 2d array (compare it with N
    L2 norms on the columns of the 2d array
    """
    l21 = L21(ndim=2, sigma=par['sigma'])

    # norm
    x = np.random.normal(0., 1., 2 * par['nx']).astype(par['dtype'])
    l21_ = par['sigma'] * np.sum(np.linalg.norm(x.reshape(2, par['nx']),
                                                axis=0))
    assert_array_almost_equal(l21(x), l21_)

    # prox / dualprox
    tau = 2.
    assert moreau(l21, x, tau)
