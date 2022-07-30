import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pylops import MatrixMult, Identity

import pyproximal
from pyproximal.utils import moreau
from pyproximal.proximal import L1, L2, Nonlinear, Orthogonal, Quadratic, \
    SingularValuePenalty, VStack, QuadraticEnvelopeCardIndicator, \
    QuadraticEnvelopeRankL2

par1 = {'nx': 10, 'ny': 10, 'sigma': 1., 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'ny': 14, 'sigma': 2., 'dtype': 'float64'}  # odd float64


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


@pytest.mark.parametrize("par,expected", [(par1, 94.89988856174841), (par2, 145.6421905545182)])
def test_QuadraticEnvelopeCardIndicator_case01(par, expected):
    """QuadraticEnvelopeCardIndicator penalty and proximal/dual proximal
    """
    np.random.seed(10)
    fr0 = QuadraticEnvelopeCardIndicator(4)
    # Quadratic envelope of the indicator function of the l0-penalty
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])

    # Check value
    assert fr0(x) == pytest.approx(expected)

    # Check proximal operator
    tau = 0.35
    assert moreau(fr0, x, tau)


def test_QuadraticEnvelopeCardIndicator_case02():
    """QuadraticEnvelopeCardIndicator penalty and proximal/dual proximal
    """
    fr0 = QuadraticEnvelopeCardIndicator(5)
    # Quadratic envelope of the indicator function of the l0-penalty
    x = np.array([1, 1.5, 1.3, 4.1, 2.1, 1.6, 1.8, 1.8])

    # Check value
    assert fr0(x) == pytest.approx(6.206249999999999)

    # Check proximal operator
    tau = 0.75
    expected = np.array([0, 0.68571429, 0, 4.1, 2.1, 1.08571429, 1.8, 1.8])
    np.testing.assert_array_almost_equal(fr0.prox(x, tau), expected)


def test_QuadraticEnvelopeCardIndicator_case03():
    """QuadraticEnvelopeCardIndicator penalty and proximal/dual proximal
    """
    fr0 = QuadraticEnvelopeCardIndicator(4)
    # Quadratic envelope of the indicator function of the l0-penalty
    x = np.array([1, -1, 1, -0.1, 1])

    # Check value
    assert fr0(x) == pytest.approx(0.09624999999999995)

    # Check proximal operator
    tau = 0.75
    expected = np.array([1, -1, 1, 0, 1])
    np.testing.assert_array_almost_equal(fr0.prox(x, tau), expected)


@pytest.mark.parametrize("par,expected", [(par1, 5.931525368112523), (par2, 14.146478354862971)])
def test_QuadraticEnvelopeRankL2(par, expected):
    """QuadraticEnvelopeRankL2 penalty and proximal/dual proximal
    """
    np.random.seed(10)
    dim = (par['nx'], par['ny'])

    # Quadratic envelope of the indicator function of rank penalty
    r0 = 8
    f = QuadraticEnvelopeRankL2(dim, r0, np.zeros(dim))
    X = np.random.normal(0, 1, dim).astype(par["dtype"])

    # Check value
    assert f(X.ravel()) == pytest.approx(expected)

    # Check proximal operator
    tau = 0.75
    assert moreau(f, X.ravel(), tau)


