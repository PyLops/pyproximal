import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal

from pylops.basicoperators import MatrixMult
from pyproximal.proximal import L2
from pyproximal.utils.bilinear import LowRankFactorizedMatrix
from pyproximal.utils.gradtest import gradtest_proximal, gradtest_bilinear

par1 = {'nx': 10, 'imag': 0, 'complexflag': False, 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'imag': 0, 'complexflag': False, 'dtype': 'float64'}  # odd float64
par1j = {'nx': 10, 'imag': 1j, 'complexflag': True, 'dtype': 'complex64'}  # even complex64
par2j = {'nx': 11, 'imag': 1j, 'complexflag': True, 'dtype': 'float64'}  # odd complex128

ngrads = 20 # number of gradient tests

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_l2(par):
    """L2 gradient
    """

    # x
    l2 = L2()
    for _ in range(ngrads):
        gradtest_proximal(l2, par['nx'],
                          delta=1e-6, complexflag=par['complexflag'],
                          raiseerror=True, atol=1e-3,
                          verb=False)

    # x - b
    b = np.ones(par['nx'], dtype=par['dtype'])
    l2 = L2(b=b)
    for _ in range(ngrads):
        gradtest_proximal(l2, par['nx'],
                          delta=1e-6, complexflag=par['complexflag'],
                          raiseerror=True, atol=1e-3,
                          verb=False)

    # Opx - b
    Op = MatrixMult(np.random.normal(0, 1, (2 * par['nx'], par['nx'])) +
                    par['imag'] * np.random.normal(0, 1, (2 * par['nx'], par['nx'])),
                    dtype=par['dtype'])
    b = np.ones(2 * par['nx'], dtype=par['dtype'])
    l2 = L2(b=b, Op=Op)
    for _ in range(ngrads):
        gradtest_proximal(l2, par['nx'],
                          delta=1e-6, complexflag=par['complexflag'],
                          raiseerror=True, atol=1e-3,
                          verb=False)

@pytest.mark.parametrize("par", [(par1), (par2), (par1j), (par2j)])
def test_lowrank(par):
    """LowRankFactorizedMatrix gradient
    """
    n, m, k = 2 * par['nx'], par['nx'], par['nx'] // 2
    x = np.random.normal(0, 1, (n, k)) + par['imag'] * np.random.normal(0, 1, (n, k))
    y = np.random.normal(0, 1, (k, m)) + par['imag'] * np.random.normal(0, 1, (k, m))
    d = np.random.normal(0, 1, (n, m)) + par['imag'] * np.random.normal(0, 1, (n, m))

    hop = LowRankFactorizedMatrix(x.copy(), y.copy(), d.ravel())

    for _ in range(ngrads):
        gradtest_bilinear(hop, delta=1e-6, complexflag=par['complexflag'],
                          raiseerror=True, atol=1e-3,
                          verb=False)