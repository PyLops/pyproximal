import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import Identity, MatrixMult

from pyproximal.optimization.primal import (
    ADMM,
    ADMML2,
    HQS,
    DouglasRachfordSplitting,
    GeneralizedProximalGradient,
    LinearizedADMM,
    ProximalGradient,
)
from pyproximal.optimization.primaldual import PrimalDual
from pyproximal.proximal import L1, L2

par1 = {"n": 8, "m": 10, "dtype": "float32"}  # float64
par2 = {"n": 8, "m": 10, "dtype": "float64"}  # float32


def test_HQS_noinitial():
    """Check that an error is raised if no initial value
    is provided to HQS solver
    """
    with pytest.raises(ValueError):
        _ = HQS(
            proxf=L2(),
            proxg=L1(),
            tau=1.0,
            x0=None,
            z0=None,
        )


def test_ADMM_noinitial():
    """Check that an error is raised if no initial value
    is provided to ADMM solver
    """
    with pytest.raises(ValueError):
        _ = ADMM(
            proxf=L2(),
            proxg=L1(),
            tau=1.0,
            x0=None,
            z0=None,
        )


def test_ADMML2_noinitial():
    """Check that an error is raised if no initial x0
    is provided to PrimalDual solver
    """
    with pytest.raises(ValueError):
        _ = ADMML2(
            proxg=L1(),
            Op=Identity(10),
            b=np.ones(10),
            A=Identity(10),
            x0=None,
            tau=1.0,
        )


def test_LinearizedADMM_noinitial():
    """Check that an error is raised if no initial x0
    is provided to LinearizedADMM solver
    """
    with pytest.raises(ValueError):
        _ = LinearizedADMM(
            proxf=L2(),
            proxg=L1(),
            A=Identity(10),
            x0=None,
            tau=1.0,
            mu=1.0,
        )


def test_PrimalDual_noinitial():
    """Check that an error is raised if no initial x0
    is provided to PrimalDual solver
    """
    with pytest.raises(ValueError):
        _ = PrimalDual(
            proxf=L2(),
            proxg=L1(),
            A=Identity(10),
            x0=None,
            tau=1.0,
            mu=1.0,
        )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_GPG_weights(par):
    """Check GPG raises error if weight is not summing to 1"""
    with pytest.raises(ValueError):
        np.random.seed(0)
        n, m = par["n"], par["m"]

        # Random mixing matrix
        R = np.random.normal(0.0, 1.0, (n, m))
        Rop = MatrixMult(R)

        # Model and data
        x = np.zeros(m)
        y = Rop @ x

        # Operators
        l2 = L2(Op=Rop, b=y, niter=10, warm=True)
        l1 = L1(sigma=5e-1)
        _ = GeneralizedProximalGradient(
            [
                l2,
            ],
            [
                l1,
            ],
            x0=np.zeros(m),
            tau=1.0,
            weights=[1.0, 1.0],
        )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PG_GPG(par):
    """Check equivalency of ProximalGradient and GeneralizedProximalGradient when using
    a single regularization term
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real
    tau = 0.99 / L

    # PG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xpg = ProximalGradient(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, acceleration="fista"
    )

    # GPG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xgpg = GeneralizedProximalGradient(
        [
            l2,
        ],
        [
            l1,
        ],
        x0=np.zeros(m),
        tau=tau,
        niter=100,
        acceleration="fista",
    )

    assert_array_almost_equal(xpg, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ADMM_DRS(par):
    """Check equivalency of ADMM and DouglasRachfordSplitting
    when using a single regularization term
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m))
    Rop = MatrixMult(R)

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    # ADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xadmm = ADMM(l2, l1, x0=np.zeros(m), tau=tau, niter=100, show=True)

    # DRS with g first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_g = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, show=True, gfirst=True
    )

    # DRS with f first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_f = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, show=True, gfirst=False
    )

    assert_array_almost_equal(xadmm, xdrs_g, decimal=2)
    assert_array_almost_equal(xadmm, xdrs_f, decimal=2)
