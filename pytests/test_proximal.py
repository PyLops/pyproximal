import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops import MatrixMult, Identity
from pyproximal.utils import moreau
from pyproximal.proximal import Quadratic, L1, L2, Orthogonal, VStack

par1 = {'nx': 10, 'sigma': 1., 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'sigma': 2., 'dtype': 'float64'}  # odd float64

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Quadratic(par):
    """Quadratic functional and proximal/dual proximal
    """
    A = np.random.normal(0, 1, (par['nx'], par['nx']))
    A = A.T @ A
    quad = Quadratic(Op=MatrixMult(A), b=np.ones(par['nx']), niter=500)

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(quad, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_DotProduct(par):
    """Dot product functional and proximal/dual proximal
    """
    quad = Quadratic(b=np.ones(par['nx']))

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(quad, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Constant(par):
    """Constant functional and proximal/dual proximal
    """
    quad = Quadratic(c=5.)

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(quad, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_SemiOrthogonal(par):
    """L1 functional with Semi-Orthogonal operator and proximal/dual proximal
    """
    l1 = L1()
    orth = Orthogonal(l1, 2*Identity(par['nx']), b=np.arange(par['nx']),
                      partial=True, alpha=4.)

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(orth, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Orthogonal(par):
    """L1 functional with Orthogonal operator and proximal/dual proximal
    """
    l1 = L1()
    orth = Orthogonal(l1, Identity(par['nx']), b=np.arange(par['nx']))

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(orth, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_VStack(par):
    """L2 functional with VStack operator of multiple L1s
    """
    nxs = [par['nx'] // 4] * 4
    nxs[-1] = par['nx'] - np.sum(nxs[:-1])
    l2 = L2()
    vstack = VStack([l2] * 4, nxs)

    # functional
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert_array_almost_equal(l2(x), vstack(x), decimal=4)

    # gradient
    assert_array_almost_equal(l2.grad(x), vstack.grad(x), decimal=4)

    # prox / dualprox
    tau = 2.
    assert_array_equal(l2.prox(x, tau), vstack.prox(x, tau))

    # moreau
    assert moreau(vstack, x, tau)
