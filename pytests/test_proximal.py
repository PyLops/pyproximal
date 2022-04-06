import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops import MatrixMult, Identity

import pyproximal
from pyproximal.utils import moreau
from pyproximal.proximal import L1, L2, Nonlinear, Orthogonal, Quadratic, \
    SingularValuePenalty, VStack

par1 = {'nx': 10, 'sigma': 1., 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'sigma': 2., 'dtype': 'float64'}  # odd float64


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Quadratic(par):
    """Quadratic functional and proximal/dual proximal
    """
    np.random.seed(10)
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
    np.random.seed(10)
    quad = Quadratic(b=np.ones(par['nx']))

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(quad, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Constant(par):
    """Constant functional and proximal/dual proximal
    """
    np.random.seed(10)
    quad = Quadratic(c=5.)

    # prox / dualprox
    tau = 2.
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert moreau(quad, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_SemiOrthogonal(par):
    """L1 functional with Semi-Orthogonal operator and proximal/dual proximal
    """
    np.random.seed(10)
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
    np.random.seed(10)
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
    np.random.seed(10)
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


def test_Nonlinear():
    """Nonlinear proximal operator. Since this is a template class simply check
    that errors are raised when not used properly
    """
    np.random.seed(10)
    Nop = Nonlinear(np.ones(10))
    with pytest.raises(NotImplementedError):
        Nop.fun(np.ones(10))
    with pytest.raises(NotImplementedError):
        Nop.grad(np.ones(10))
    with pytest.raises(NotImplementedError):
        Nop.optimize()


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_SingularValuePenalty(par):
    """Test SingularValuePenalty
    """
    np.random.seed(10)
    f_mu = pyproximal.QuadraticEnvelopeCard(mu=par['sigma'])
    penalty = SingularValuePenalty((par['nx'], 2 * par['nx']), f_mu)

    # norm, cross-check with svd (use tolerance as two methods don't provide
    # the exact same eigenvalues)
    X = np.random.uniform(0., 0.1, (par['nx'], 2 * par['nx'])).astype(par['dtype'])
    _, S, _ = np.linalg.svd(X)
    assert (penalty(X.ravel()) - f_mu(S)) < 1e-3

    # prox / dualprox
    tau = 0.75
    assert moreau(penalty, X.ravel(), tau)
