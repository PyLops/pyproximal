import pytest

import numpy as np
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import MatrixMult
from pyproximal.proximal import L1, L2
from pyproximal.optimization.primal import (
    ProximalGradient,
    GeneralizedProximalGradient,
    ADMM,
    DouglasRachfordSplitting,
)

par1 = {'n': 8, 'm': 10,  'dtype': 'float32'}  # float64
par2 = {'n': 8, 'm': 10,  'dtype': 'float64'}  # float32


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_GPG_weights(par):
    """Check GPG raises error if weight is not summing to 1
    """
    with pytest.raises(ValueError):
        np.random.seed(0)
        n, m = par['n'], par['m']

        # Random mixing matrix
        R = np.random.normal(0., 1., (n, m))
        Rop = MatrixMult(R)

        # Model and data
        x = np.zeros(m)
        y = Rop @ x

        # Operators
        l2 = L2(Op=Rop, b=y, niter=10, warm=True)
        l1 = L1(sigma=5e-1)
        _ = GeneralizedProximalGradient([l2, ], [l1, ],
                                        x0=np.zeros(m),
                                        tau=1.,
                                        weights=[1., 1.])


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PG_GPG(par):
    """Check equivalency of ProximalGradient and GeneralizedProximalGradient when using
    a single regularization term
    """
    np.random.seed(0)
    n, m = par['n'], par['m']

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = np.random.normal(0., 1., (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real
    tau = 0.99 / L

    # PG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xpg = ProximalGradient(l2, l1, x0=np.zeros(m),
                           tau=tau, niter=100,
                           acceleration='fista')

    # GPG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xgpg = GeneralizedProximalGradient([l2, ], [l1, ], x0=np.zeros(m),
                                       tau=tau, niter=100,
                                       acceleration='fista')

    assert_array_almost_equal(xpg, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ADMM_DRS(par):
    """Check equivalency of ADMM and DouglasRachfordSplitting
    when using a single regularization term
    """
    np.random.seed(0)
    n, m = par['n'], par['m']

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = np.random.normal(0., 1., (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    # PG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xadmm = ADMM(
        l2, l1, x0=np.zeros(m),
        tau=tau, niter=100, show=True
    )

    # DRS
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_g = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m),
        tau=tau, niter=100, show=True,
        gfirst=True
    )

    # DRS
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_f = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m),
        tau=tau, niter=100, show=True,
        gfirst=False
    )

    assert_array_almost_equal(xadmm, xdrs_g, decimal=2)
    assert_array_almost_equal(xadmm, xdrs_f, decimal=2)
