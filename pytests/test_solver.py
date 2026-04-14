from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import Identity, MatrixMult

from pyproximal.optimization.primal import (
    ADMM,
    ADMML2,
    HQS,
    PPXA,
    ConsensusADMM,
    DouglasRachfordSplitting,
    GeneralizedProximalGradient,
    LinearizedADMM,
    ProximalGradient,
    ProximalPoint,
)
from pyproximal.proximal import L1, L2, Quadratic

par1 = {"n": 8, "m": 10, "dtype": "float32"}  # float64
par2 = {"n": 8, "m": 10, "dtype": "float64"}  # float32


def test_HQS_noinitial():
    """Check that an error is raised if no initial value
    is provided to HQS solver
    """
    with pytest.raises(ValueError, match="Both x0 or "):
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
    with pytest.raises(ValueError, match="Both x0 or"):
        _ = ADMM(
            proxf=L2(),
            proxg=L1(),
            tau=1.0,
            x0=None,
            z0=None,
        )


def test_ADMML2_noinitial():
    """Check that an error is raised if no initial value
    is provided to PrimalDual solver
    """
    with pytest.raises(ValueError, match="Both x0 or"):
        # Both None
        _ = ADMML2(
            proxg=L1(),
            Op=Identity(10),
            b=np.ones(10),
            A=Identity(10),
            tau=1.0,
            x0=None,
            z0=None,
        )
    with pytest.raises(ValueError, match="x0 must be provided when"):
        # x0 is None (and Op provided)
        _ = ADMML2(
            proxg=L1(),
            Op=Identity(10),
            b=np.ones(10),
            A=Identity(10),
            tau=1.0,
            x0=None,
            z0=np.ones(10),
        )


def test_LinearizedADMM_noinitial():
    """Check that an error is raised if no initial x0
    is provided to LinearizedADMM solver
    """
    with pytest.raises(ValueError, match="Both x0 or "):
        # Both None
        _ = LinearizedADMM(
            proxf=L2(),
            proxg=L1(),
            A=Identity(10),
            tau=1.0,
            mu=1.0,
            x0=None,
            z0=None,
        )
    with pytest.raises(ValueError, match="x0 must be provided when"):
        # x0 is None (and Op provided)
        _ = LinearizedADMM(
            proxf=L2(),
            proxg=L1(),
            A=Identity(10),
            tau=1.0,
            mu=1.0,
            x0=None,
            z0=np.ones(10),
        )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_GPG_weights(par):
    """Check GPG raises error if weight is not summing to 1"""
    with pytest.raises(ValueError, match="must be an array of size"):
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
def test_ProximalPoint(par):
    """Check solution of ProximalPoint for quadratic function equals the solution of the
    associated system of linear equations
    """
    np.random.seed(10)
    m = par["m"]

    # Random mixing matrix
    A = np.random.normal(0.0, 1.0, (m, m))
    A = A.T @ A

    # Model and data
    x = np.linspace(-5.0, 5.0, par["m"])
    y = A @ x

    # Proximal point algorithm with quadatic function
    quad = Quadratic(Op=MatrixMult(A), b=-y, niter=2)
    xpp = ProximalPoint(quad, x0=np.zeros_like(x), tau=0.1, niter=1000, tol=0)

    assert_array_almost_equal(xpp, x, decimal=2)


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
    xadmm, zadmm = ADMM(l2, l1, x0=np.zeros(m), tau=tau, niter=100)

    # DRS with g first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_g, ydrs_g = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, gfirst=True
    )

    # DRS with f first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_f, ydrs_f = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, gfirst=False
    )

    assert_array_almost_equal(xadmm, xdrs_g, decimal=2)
    assert_array_almost_equal(xadmm, xdrs_f, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PPXA_with_ADMM(par: dict[str, Any]) -> None:
    """Check equivalency of PPXA and ADMM
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

    l2 = L2(Op=Rop, b=y, niter=50, warm=False)
    l1 = L1(sigma=5e-1)

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    xadmm, _ = ADMM(
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=2000,  # niter=1500 makes this test fail for seeds 0 to 499
    )
    xppxa = PPXA(
        [l2, l1],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
    )

    assert_array_almost_equal(xppxa, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_PPXA_with_GPG(par: dict[str, Any]) -> None:
    """Check equivalency of PPXA and GeneralizedProximalGradient"""
    np.random.seed(0)

    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    g = np.zeros_like(x)
    g[1], g[2] = 1, 0.5

    # Random mixing matrices
    R1 = np.random.normal(0.0, 1.0, (n, m))
    Rop1 = MatrixMult(R1)
    y1 = Rop1 @ x

    R2 = np.random.normal(0.0, 1.0, (n, m))
    Rop2 = MatrixMult(R2)
    y2 = Rop2 @ x

    l2_1 = L2(Op=Rop1, b=y1, niter=50, warm=False)
    l2_2 = L2(Op=Rop2, b=y2, niter=50, warm=False)
    l1_1 = L1(sigma=5e-1)
    l1_2 = L1(sigma=2.5e-1, g=g)

    # Step size
    L = (Rop1.H * Rop1).eigs(1).real.item()
    tau = 0.5 / L

    xgpg = GeneralizedProximalGradient(
        [l2_1, l2_2],
        [l1_1, l1_2],
        x0=np.zeros(m),
        tau=tau,
        niter=200,  # niter=150 makes this test fail for seeds 0 to 499
    )
    xppxa = PPXA(
        [l2_1, l2_2, l1_1, l1_2],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
    )

    assert_array_almost_equal(xppxa, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ConsensusADMM_with_ADMM(par: dict[str, Any]) -> None:
    """Check equivalency of ConsensusADMM and ADMM
    when two proximable functions
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

    l2 = L2(Op=Rop, b=y, niter=50, warm=False)
    l1 = L1(sigma=5e-1)

    # Step size
    L = (Rop.H * Rop).eigs(1).real.item()
    tau = 0.5 / L

    xadmm, _ = ADMM(
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=2000,  # niter=1500 makes this test fail for seeds 0 to 499
    )
    xcadmm = ConsensusADMM(
        [l2, l1],
        x0=np.random.normal(0.0, 1.0, m),  # x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
    )

    assert_array_almost_equal(xcadmm, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ConsensusADMM_with_ADMM_for_Lasso(par: dict[str, Any]) -> None:
    """Check equivalency of ConsensusADMM and ADMM
    when more than two proximable functions for lasso
    """
    m = par["m"]
    lmd = 1e-2
    n_l2_ops = 3

    np.random.seed(0)

    # Define sparse model
    x_true = np.zeros(m)
    nnz = np.random.randint(3, m // 2)
    support = np.random.choice(m, size=nnz, replace=False)
    x_true[support] = np.random.normal(0.0, 1.0, size=len(support))

    # Random mixing matrix
    R_list, y_list = [], []
    for ni in np.random.randint(3, 10, size=n_l2_ops):
        R = np.random.normal(0.0, 1.0, size=(ni, m))
        R_list.append(R)
        y_list.append(R @ x_true)

    # 1/2||R1||_2^2, 1/2||R2||_2^2, 1/2||R3||_2^2
    l2_ops = [
        L2(Op=MatrixMult(Ri), b=yi, niter=50, warm=False)
        for Ri, yi in zip(R_list, y_list, strict=True)
    ]

    # 1/2 || [R1; R2; R3] ||_2^2
    Rop_stack = MatrixMult(np.vstack(R_list))
    y_stack = np.concatenate(y_list)
    l2_stack = L2(Op=Rop_stack, b=y_stack, niter=50, warm=False)

    # ||x||_1
    l1_op = L1(sigma=lmd)

    # Step size
    L = (Rop_stack.H * Rop_stack).eigs(1).real.item()
    tau = 0.5 / L

    # 1/2||R1||_2^2 + 1/2||R2||_2^2 + 1/2||R3||_2^2 + ||x||_1
    xcadmm = ConsensusADMM(
        [*l2_ops, l1_op],
        x0=np.random.normal(0.0, 1.0, m),  # x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
        niter=20000,  # niter=15000 makes this test fail for seeds 0 to 499
    )

    # 1/2 || [R1; R2; R3] ||_2^2 + ||x||_1
    xadmm, _ = ADMM(
        l2_stack,
        l1_op,
        x0=np.zeros(m),
        tau=tau,
        niter=15000,  # niter=10000 makes this test fail for seeds 0 to 499
    )

    assert_array_almost_equal(xcadmm, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ConsensusADMM_with_GPG(par: dict[str, Any]) -> None:
    """Check equivalency of ConsensusADMM and GeneralizedProximalGradient"""

    np.random.seed(0)

    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m)
    x[2], x[4] = 1, 0.5

    g = np.zeros_like(x)
    g[1], g[2] = 1, 0.5

    # Random mixing matrices
    R1 = np.random.normal(0.0, 1.0, (n, m))
    Rop1 = MatrixMult(R1)
    y1 = Rop1 @ x

    R2 = np.random.normal(0.0, 1.0, (n, m))
    Rop2 = MatrixMult(R2)
    y2 = Rop2 @ x

    l2_1 = L2(Op=Rop1, b=y1, niter=50, warm=False)
    l2_2 = L2(Op=Rop2, b=y2, niter=50, warm=False)
    l1_1 = L1(sigma=5e-1)
    l1_2 = L1(sigma=2.5e-1, g=g)

    # Step size
    L = (Rop1.H * Rop1).eigs(1).real.item()
    tau = 0.5 / L

    xgpg = GeneralizedProximalGradient(
        [l2_1, l2_2],
        [l1_1, l1_2],
        x0=np.zeros(m),
        tau=tau,
        niter=200,  # niter=150 makes this test fail for seeds 0 to 499
    )
    xppxa = ConsensusADMM(
        [l2_1, l2_2, l1_1, l1_2],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
    )

    assert_array_almost_equal(xppxa, xgpg, decimal=2)
