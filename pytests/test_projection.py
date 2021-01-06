import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pyproximal.utils import moreau
from pyproximal.proximal import Box, Simplex

par1 = {'nx': 10, 'ny': 8, 'axis': 0, 'dtype': 'float32'}  # even float32 dir0
par2 = {'nx': 11, 'ny': 8, 'axis': 1, 'dtype': 'float64'}  # odd float64 dir1
par3 = {'nx': 10, 'ny': 8, 'axis': 0, 'dtype': 'float32'}  # even float32 dir0
par4 = {'nx': 11, 'ny': 8, 'axis': 1, 'dtype': 'float64'}  # odd float64  dir1

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Box(par):
    """Box projection and proximal/dual proximal of related indicator
    """
    box = Box(-1, 1)
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])

    # prox / dualprox
    tau = 2.
    assert moreau(box, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Simplex(par):
    """Simplex projection and proximal/dual proximal of related indicator
    """
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
