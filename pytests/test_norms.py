import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import Identity, Diagonal, MatrixMult, FirstDerivative
from pyproximal.utils import moreau
from pyproximal.proximal import Box, Euclidean, L2, L1, L21, L21_plus_L1, Huber, Nuclear, TV

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
    """L2 norm of Op*x-b and proximal/dual proximal
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
    """L2 norm of difference (x-b) and proximal/dual proximal
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
def test_L2_dense(par):
    """L2 norm of Op*x with dense Op and proximal/dual proximal
    """
    for densesolver in ('numpy', 'scipy', 'factorize'):
        b = np.zeros(par['nx'], dtype=par['dtype'])
        d = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
        l2 = L2(Op=MatrixMult(np.diag(d), dtype=par['dtype']),
                b=b, sigma=par['sigma'], densesolver=densesolver)

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


@pytest.mark.parametrize('par', [(par1), (par2)])
def test_L21_plus_L1(par):
    """L21 plus L1 norm on 2darray.
    """
    l21_plus_l1 = L21_plus_L1(sigma=par['sigma'], rho=0.8)

    # norm
    x = np.random.normal(0., 1., (2 * par['nx'], par['nx'])).astype(par['dtype'])
    rho = 0.8
    l21_plus_l1_ = par['sigma'] * rho * np.sum(np.abs(x)) + par['sigma'] * (1 - rho) * np.sum(
        np.sqrt(np.sum(x ** 2, axis=0))
    )
    assert_array_almost_equal(l21_plus_l1(x), l21_plus_l1_)

    tau = 2.
    l21_plus_l1.prox(x, tau)

    assert moreau(l21_plus_l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Huber(par):
    """Huber norm and proximal/dual proximal
    """
    hub = Huber(alpha=par['sigma'])

    # norm
    x = np.random.uniform(0., 0.1, par['nx']).astype(par['dtype'])
    assert hub(x) == np.linalg.norm(x) ** 2 / (2 * par['sigma'])

    # prox / dualprox
    tau = 2.
    assert moreau(hub, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_TV(par):
    """TV norm of x and proximal
    """
    tv = TV(dims=(par['nx'], ), sigma=par['sigma'])
    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    derivOp = FirstDerivative(par['nx'], dtype=par['dtype'], kind='forward')
    dx = derivOp @ x
    assert_array_almost_equal(tv(x), par['sigma'] * np.sum(np.abs(dx), axis=0))


def test_Nuclear_FOM():
    """Nuclear norm benchmark with FOM solver
    """
    X = np.array([[1, 2, 3],
                  [4, 5, 6]])
    proxX = np.array([[1.4089, 1.8613, 2.3137],
                      [3.3640, 4.4441, 5.5242]])
    nucl = Nuclear(X.shape)
    assert_array_almost_equal(proxX.ravel(), nucl.prox(X, 1), decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Nuclear(par):
    """Nuclear norm and proximal/dual proximal
    """
    nucl = Nuclear((par['nx'], 2 * par['nx']), sigma=par['sigma'])

    # norm, cross-check with svd (use tolerance as two methods don't provide
    # the exact same eigenvalues)
    X = np.random.uniform(0., 0.1, (par['nx'], 2 * par['nx'])).astype(par['dtype'])
    _, S, _ = np.linalg.svd(X)
    assert (nucl(X.ravel()) - par['sigma'] * np.sum(S)) < 1e-3

    # prox / dualprox
    tau = 2.
    assert moreau(nucl, X.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Weighted_Nuclear(par):
    """Weighted nuclear norm and proximal/dual proximal
    """
    weights = par['sigma'] * np.linspace(0.1, 5, 2 * par['nx'])
    nucl = Nuclear((par['nx'], 2 * par['nx']), sigma=weights)

    # norm, cross-check with svd (use tolerance as two methods don't provide
    # the exact same singular values)
    X = np.random.uniform(0., 0.1, (par['nx'], 2 * par['nx'])).astype(par['dtype'])
    S = np.linalg.svd(X, compute_uv=False)
    assert (nucl(X.ravel()) - np.sum(weights[:S.size] * S)) < 1e-2

    # prox / dualprox
    tau = 2.
    assert moreau(nucl, X.ravel(), tau)
