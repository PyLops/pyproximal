import pytest

import numpy as np
from numpy.testing import assert_array_equal
from pylops import Diagonal

from pyproximal.utils.bilinear import LowRankFactorizedMatrix


par1 = {'n': 21, 'm': 11, 'k': 5, 'imag': 0,
        'dtype': 'float32'}  # real
par2 = {'n': 21, 'm': 11, 'k': 5, 'imag': 1j,
        'dtype': 'complex64'}  # complex

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_lrfactorized(par):
    """Check equivalence of matvec operations in LowRankFactorizedMatrix
    """
    U = (np.random.normal(0., 1., (par['n'], par['k'])) + \
        par['imag'] * np.random.normal(0., 1., (par['n'], par['k']))).astype(par['dtype'])
    V = (np.random.normal(0., 1., (par['m'], par['k'])) + \
         par['imag'] * np.random.normal(0., 1., (par['m'], par['k']))).astype(par['dtype'])

    X = U @ V.T
    LOp = LowRankFactorizedMatrix(U, V.T, X)

    assert_array_equal(X.ravel(), LOp._matvecx(U.ravel()))
    assert_array_equal(X.ravel(), LOp._matvecy(V.T.ravel()))


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_lrfactorizedoperator(par):
    """Check equivalence of matvec operations in LowRankFactorizedMatrix with operator Op
    """
    U = (np.random.normal(0., 1., (par['n'], par['k'])) + \
        par['imag'] * np.random.normal(0., 1., (par['n'], par['k']))).astype(par['dtype'])
    V = (np.random.normal(0., 1., (par['m'], par['k'])) + \
         par['imag'] * np.random.normal(0., 1., (par['m'], par['k']))).astype(par['dtype'])
    Op = Diagonal(np.arange(par['n'] * par['m']) + 1.)

    X = U @ V.T
    y = Op @ X.ravel()

    LOp = LowRankFactorizedMatrix(U, V.T, y, Op)

    assert_array_equal(y, LOp._matvecx(U.ravel()))
    assert_array_equal(y, LOp._matvecy(V.T.ravel()))
