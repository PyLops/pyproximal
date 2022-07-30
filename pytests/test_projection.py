import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import Identity
from pyproximal.utils import moreau
from pyproximal.proximal import Box, EuclideanBall, L0Ball, L1Ball, \
    NuclearBall, Simplex, AffineSet, Hankel

par1 = {'nx': 10, 'ny': 8, 'axis': 0, 'dtype': 'float32'}  # even float32 dir0
par2 = {'nx': 11, 'ny': 8, 'axis': 1, 'dtype': 'float64'}  # odd float64 dir1
par3 = {'nx': 10, 'ny': 8, 'axis': 0, 'dtype': 'float32'}  # even float32 dir0
par4 = {'nx': 11, 'ny': 8, 'axis': 1, 'dtype': 'float64'}  # odd float64  dir1


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Box(par):
    """Box projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    box = Box(-1, 1)
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])

    # prox / dualprox
    tau = 2.
    assert moreau(box, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_EuclBall(par):
    """Euclidean Ball projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    eucl = EuclideanBall(np.zeros(par['nx']), 1.)
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype']) + 1.

    # evaluation
    assert eucl(x) == False
    xp = eucl.prox(x, 1.)
    assert eucl(xp) == True

    # prox / dualprox
    tau = 2.
    assert moreau(eucl, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L0Ball(par):
    """L0 Ball projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    l0 = L0Ball(1)
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype']) + 1.

    # evaluation
    assert l0(x) == False
    xp = l0.prox(x, 1.)
    assert l0(xp) == True

    # prox / dualprox
    tau = 2.
    assert moreau(l0, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1Ball(par):
    """L1 Ball projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    l1 = L1Ball(par['nx'], 1)
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype']) + 1.

    # evaluation
    assert l1(x) == False
    xp = l1.prox(x, 1.)
    assert l1(xp) == True

    # prox / dualprox
    tau = 2.
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_NuclBall(par):
    """Nuclear Ball projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    nuc = NuclearBall((par['nx'], par['ny']), 1)
    x = np.random.normal(0., 1., (par['nx'], par['ny'])).astype(par['dtype']) \
        + np.eye(par['nx'], par['ny'])

    # evaluation
    assert nuc(x) == False
    xp = nuc.prox(x, 1.)
    assert nuc(xp, 1e-4) == True

    # prox / dualprox
    tau = 2.
    assert moreau(nuc, x.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Simplex(par):
    """Simplex projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    for engine in ['numpy', 'numba']:
        x = np.abs(np.random.normal(0., 1., par['nx']).astype(par['dtype']))

        sim = Simplex(n=par['nx'], radius=np.sum(x), engine=engine)
        sim1 = Simplex(n=par['nx'], radius=np.sum(x) - 0.1, engine=engine)

        # evaluation
        assert sim(x) == True
        assert sim1(x) == False

        # prox / dualprox
        tau = 2.
        assert moreau(sim, x, tau)
        assert moreau(sim1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Simplex_multi(par):
    """Simplex projection and proximal/dual proximal for 2d array
    """
    np.random.seed(10)

    dims = (par['ny'], par['nx'])
    otheraxis = 1 if par['axis'] == 0 else 0
    for engine in ['numpy', 'numba']:
        x = np.abs(np.random.normal(0., 1., dims).astype(par['dtype']))

        sim = Simplex(n=par['ny'] * par['nx'], radius=5., dims=dims,
                      axis=par['axis'], maxiter=50, engine=engine)

        # prox
        tau = 2.
        y = sim.prox(x.ravel(), tau)
        assert_array_almost_equal(np.sum(y.reshape(dims), axis=par['axis']),
                                  5. * np.ones(dims[otheraxis]), decimal=1)

        # prox / dualprox
        assert moreau(sim, x.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2), (par3), (par4)])
def test_Affine(par):
    """Affine set projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)

    Op = Identity(par['nx'])
    b = np.random.normal(0., 1., par['nx'])
    aff = AffineSet(Op, b, 10)

    # prox
    tau = 2.
    x = np.ones(par['nx'])
    assert_array_almost_equal(aff.prox(x, tau), b)

    # prox / dualprox
    assert moreau(aff, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Hankel(par):
    """Hankel matrix projection and proximal/dual proximal of related indicator
    """
    np.random.seed(10)
    dim = (par['nx'], par['ny'])
    hankel = Hankel(dim)
    x = np.random.normal(0., 1., dim).astype(par['dtype'])

    # evaluation
    assert hankel(x) == False
    xp = hankel.prox(x, 1.)
    assert hankel(xp) == True

    # prox / dualprox
    tau = 2.
    assert moreau(hankel, x.ravel(), tau)
