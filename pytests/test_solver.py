import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import FirstDerivative, Identity, MatrixMult
from pylops.optimization.leastsquares import regularized_inversion
from pylops.optimization.sparsity import fista, ista

from pyproximal.optimization.primal import (
    ADMM,
    ADMML2,
    HQS,
    PPXA,
    AcceleratedProximalGradient,
    AndersonProximalGradient,
    ConsensusADMM,
    DouglasRachfordSplitting,
    GeneralizedProximalGradient,
    LinearizedADMM,
    ProximalGradient,
    ProximalPoint,
    TwIST,
)
from pyproximal.optimization.primaldual import AdaptivePrimalDual, PrimalDual
from pyproximal.proximal import L1, L2, Box, Quadratic

par1 = {"n": 10, "m": 10, "dtype": "float64"}  # square, float64
par2 = {"n": 8, "m": 10, "dtype": "float64"}  # underdetermined, float64
par3 = {"n": 8, "m": 10, "dtype": "float32"}  # underdetermined, float32


def test_ProximalGradient_unknown_acceleration():
    """Check that an error is raised if an unknown acceleration
    method is provided to ProximalGradient solver
    """
    with pytest.raises(NotImplementedError, match="Acceleration should "):
        _ = ProximalGradient(proxf=L2(), proxg=L1(), x0=None, acceleration="unknown")


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
    is provided to ADMML2 solver
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


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_GPG_epsg(par):
    """Check GPG raises error if epsg is a vector with number
    of elements differring from the number of g functions
    passed to proxgs"""
    with pytest.raises(ValueError, match="must be a scalar or a vector"):
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
        box = Box()
        _ = GeneralizedProximalGradient(
            [
                l2,
            ],
            [
                l1,
                box,
            ],
            x0=np.zeros(m),
            tau=1.0,
            epsg=np.ones(5),
        )


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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
    xpp = ProximalPoint(
        quad, x0=np.zeros_like(x), tau=0.1, niter=1000, tol=0, show=True
    )

    assert_array_almost_equal(xpp, x, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PG_ISTA(par):
    """Check equivalency of ProximalGradient and ISTA/FISTA (PyLops)"""
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

    for solver, acceleration in zip(
        [ista, fista],
        [None, "fista"],
        strict=True,
    ):
        # ISTA/FISTA
        eps = 5e-1
        xista = solver(
            Rop, y, niter=100, alpha=tau, eps=eps, tol=1e-8, monitorres=False
        )[0]

        # PG
        l2 = L2(Op=Rop, b=y)
        l1 = L1()
        epsg = eps * 0.5  # to compensate for 0.5 in ISTA: thresh = eps * alpha * 0.5
        xpg = ProximalGradient(
            l2,
            l1,
            x0=np.zeros(m),
            tau=tau,
            epsg=epsg,
            acceleration=acceleration,
            niter=100,
            tol=1e-8,
            show=True,
        )

        assert_array_almost_equal(xpg, xista, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=100,
        acceleration="fista",
        show=True,
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
        show=True,
    )

    assert_array_almost_equal(xpg, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("eta", [1, 0.8])
@pytest.mark.parametrize("acceleration", [None, "fista", "vandenberghe"])
def test_PG_backtracking(par, eta, acceleration):
    """Check equivalency of ProximalGradient with and without tau (aka backtracking)"""
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

    # PG with tau
    L = (Rop.H * Rop).eigs(1).real
    tau = 0.99 / L

    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xpg = ProximalGradient(
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        eta=eta,
        niter=100,
        acceleration=acceleration,
        show=True,
    )

    # PG without tau
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xpgback = ProximalGradient(
        l2,
        l1,
        x0=np.zeros(m),
        backtracking=True,
        eta=eta,
        niter=100,
        acceleration=acceleration,
        show=True,
    )

    assert_array_almost_equal(xpg, xpgback, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PG_AcceleratedPG(par):
    """Check equivalency of ProximalGradient and AcceleratedProximalGradient"""
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
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=100,
        acceleration="fista",
        show=True,
    )

    # APG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xapg = AcceleratedProximalGradient(
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=100,
    )

    assert_array_almost_equal(xpg, xapg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PG_AndersonPG(par):
    """Check equivalency of ProximalGradient and AndersonProximalGradient"""
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
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=100,
        acceleration="fista",
        show=True,
    )

    # AndersonPG
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xapg = AndersonProximalGradient(
        l2,
        l1,
        x0=np.zeros(m),
        tau=tau,
        niter=100,
        nhistory=5,
        show=True,
    )

    assert_array_almost_equal(xpg, xapg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PG_TwiST(par):
    """Check that PG/TwiST can be used to solve a sparsity regularized objective function
    (note that despite the trajectory will be different, they should converge to the
    same solution)
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
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, acceleration="fista", show=True
    )

    # TwiST
    l1 = L1(sigma=5e-1)
    eigs = np.linalg.eig(R.T @ R)[0]
    eigs = (np.abs(eigs[0]), max(1e-1, np.abs(eigs[-1])))
    xtwist = TwIST(
        l1,
        Rop,
        y,
        x0=np.zeros(m),
        eigs=eigs,
        niter=100,
        show=True,
    )

    assert_array_almost_equal(xpg, xtwist, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("gfirst", [False, True])
def test_HQS_ADMM_L2(par, gfirst):
    """Check that HQS/ADMM can be used to solve a pure L2-based objective function
    (and compare with LSQR - note that despite the trajectory will be different,
    they should converge to the same solution)
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.random.normal(0.0, 1.0, m).astype(par["dtype"])

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m)).astype(par["dtype"])
    Rop = MatrixMult(R, dtype=par["dtype"])

    y = Rop @ x

    # Step size
    L = (Rop.H * Rop).eigs(1).real
    tau = 0.99 / L
    eps = 1e-1

    # L2
    Iop = Identity(m, dtype=par["dtype"])
    xl2 = regularized_inversion(
        Rop,
        y,
        Regs=[
            Iop,
        ],
        epsRs=[
            np.sqrt(eps),
        ],
        iter_lim=1000,
    )[0]

    # HQS
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l2reg = L2(sigma=eps)
    xhqs = HQS(
        l2, l2reg, x0=np.zeros(m), tau=tau, gfirst=gfirst, niter=1000, show=True
    )[0]

    # ADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l2reg = L2(sigma=eps)
    xadmm = ADMM(l2, l2reg, x0=np.zeros(m), tau=tau, niter=1000, show=True)[0]

    assert_array_almost_equal(xl2, xhqs, decimal=2)
    assert_array_almost_equal(xl2, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("gfirst", [False, True])
def test_ADMM_ADMML2(par, gfirst):
    """Check equivalency of ADMM and ADMML2
    when the f function is a L2 term
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.random.normal(0.0, 1.0, m).astype(par["dtype"])

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m)).astype(par["dtype"])
    Rop = MatrixMult(R, dtype=par["dtype"])

    y = Rop @ x

    # Step size
    Aop = Identity(m)
    L = 1.0  # Lipshitz constant of Aop
    tau = 0.99 / L
    eps = 1e-1

    # ADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l2reg = L2(sigma=eps)
    xadmm = ADMM(l2, l2reg, x0=np.zeros(m), tau=tau, niter=100, show=True)[0]

    # ADMML2
    l2reg = L2(sigma=eps)
    xadmml2 = ADMML2(
        l2reg,
        Rop,
        y,
        Aop,
        x0=np.zeros(m),
        tau=tau,
        gfirst=gfirst,
        niter=100,
        iter_lim=10,
        show=True,
    )[0]

    assert_array_almost_equal(xadmm, xadmml2, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_ADMM_LinearizedADMM(par):
    """Check equivalency of ADMM and LinearizedADMM
    when the f function is a L2 term
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.random.normal(0.0, 1.0, m).astype(par["dtype"])

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m)).astype(par["dtype"])
    Rop = MatrixMult(R, dtype=par["dtype"])

    y = Rop @ x

    # Step size
    Aop = Identity(m)
    L = 1.0  # Lipshitz constant of Aop
    tau = 0.99 / L
    mu = 0.99 / L  # optimal mu<=tau/maxeig(Dop^H Dop)

    eps = 1e-1

    # ADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l2reg = L2(sigma=eps)
    xadmm = ADMM(l2, l2reg, x0=np.zeros(m), tau=tau, niter=100, show=True)[0]

    # LinearizedADMM
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l2reg = L2(sigma=eps)
    Aop = Identity(m)
    xladmm = LinearizedADMM(
        l2, l2reg, Aop, x0=np.zeros(m), tau=tau, mu=mu, niter=100, show=True
    )[0]

    assert_array_almost_equal(xadmm, xladmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
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
    xadmm, zadmm = ADMM(l2, l1, x0=np.zeros(m), tau=tau, niter=100, show=True)

    # DRS with g first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_g, ydrs_g = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, gfirst=True, show=True
    )

    # DRS with f first
    l2 = L2(Op=Rop, b=y, niter=10, warm=True)
    l1 = L1(sigma=5e-1)
    xdrs_f, ydrs_f = DouglasRachfordSplitting(
        l2, l1, x0=np.zeros(m), tau=tau, niter=100, gfirst=False, show=True
    )

    assert_array_almost_equal(xadmm, xdrs_g, decimal=2)
    assert_array_almost_equal(xadmm, xdrs_f, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("weights", [None, (0.5, 0.5)])
def test_PPXA_with_ADMM(par, weights) -> None:
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
        show=True,
    )
    xppxa = PPXA(
        [l2, l1],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
        weights=weights,
        show=True,
    )

    assert_array_almost_equal(xppxa, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PPXA_with_GPG(par) -> None:
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
        show=True,
    )
    xppxa = PPXA(
        [l2_1, l2_2, l1_1, l1_2],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
        show=True,
    )

    assert_array_almost_equal(xppxa, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_ConsensusADMM_with_ADMM(par) -> None:
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
        show=True,
    )
    xcadmm = ConsensusADMM(
        [l2, l1],
        x0=np.random.normal(0.0, 1.0, m),  # x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
        show=True,
    )

    assert_array_almost_equal(xcadmm, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_ConsensusADMM_with_ADMM_for_Lasso(par) -> None:
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
        show=True,
    )

    # 1/2 || [R1; R2; R3] ||_2^2 + ||x||_1
    xadmm, _ = ADMM(
        l2_stack,
        l1_op,
        x0=np.zeros(m),
        tau=tau,
        niter=15000,  # niter=10000 makes this test fail for seeds 0 to 499
        show=True,
    )

    assert_array_almost_equal(xcadmm, xadmm, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_ConsensusADMM_with_GPG(par) -> None:
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
        show=True,
    )
    xppxa = ConsensusADMM(
        [l2_1, l2_2, l1_1, l1_2],
        x0=np.zeros(m),
        tau=np.random.uniform(3 * tau, 5 * tau),
        show=True,
    )

    assert_array_almost_equal(xppxa, xgpg, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
@pytest.mark.parametrize("gfirst", [False, True])
def test_ADMML2_PrimalDual(par, gfirst):
    """Check equivalency of ADMML2 and Primal-Dual
    (note that despite the trajectory will be different, they
    should converge to the same solution)
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m, dtype=par["dtype"])
    x[m // 2 :] = 1.0

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m)).astype(par["dtype"])
    Rop = MatrixMult(R, dtype=par["dtype"])

    y = Rop @ x

    # Step size
    Dop = FirstDerivative(m)
    L = 4.0  # Lipshitz constant of Dop
    eps = 1e-1

    # ADMML2
    l1 = L1(sigma=eps)
    tau = 0.99 / L
    xadmml2 = ADMML2(
        l1,
        Rop,
        y,
        Dop,
        x0=np.zeros(m),
        tau=tau,
        gfirst=gfirst,
        niter=200,
        iter_lim=10,
        show=True,
    )[0]

    # PD
    l2 = L2(Rop, y, niter=10, warm=True)
    l1 = L1(sigma=eps)
    tau = 0.99 / np.sqrt(L)
    mu = 0.99 / np.sqrt(L)
    xpd = PrimalDual(
        l2,
        l1,
        Dop,
        x0=np.zeros(m),
        tau=tau,
        mu=mu,
        gfirst=gfirst,
        niter=200,
        show=True,
    )

    assert_array_almost_equal(xadmml2, xpd, decimal=2)


@pytest.mark.parametrize("par", [(par1), (par2), (par3)])
def test_PrimalDual_AdaptivePrimalDual(par):
    """Check equivalency of Primal-Dual and
    Adaptive Primal-Dual (note that despite the
    trajectory will be different, they should
    converge to the same solution)
    """
    np.random.seed(0)
    n, m = par["n"], par["m"]

    # Define sparse model
    x = np.zeros(m, dtype=par["dtype"])
    x[m // 2 :] = 1.0

    # Random mixing matrix
    R = np.random.normal(0.0, 1.0, (n, m)).astype(par["dtype"])
    Rop = MatrixMult(R, dtype=par["dtype"])

    y = Rop @ x

    # Step size
    Dop = FirstDerivative(m)
    L = 4.0  # Lipshitz constant of Dop
    eps = 1e-1

    # PD
    l2 = L2(Rop, y, niter=10, warm=True)
    l1 = L1(sigma=eps)
    tau = 0.99 / np.sqrt(L)
    mu = 0.99 / np.sqrt(L)
    xpd = PrimalDual(
        l2,
        l1,
        Dop,
        x0=np.zeros(m),
        tau=tau,
        mu=mu,
        niter=200,
        show=True,
    )

    # Adaptive PD
    l2 = L2(Rop, y, niter=10, warm=True)
    l1 = L1(sigma=eps)
    tau = 0.99 / np.sqrt(L)
    mu = 0.99 / np.sqrt(L)
    xapd = AdaptivePrimalDual(
        l2,
        l1,
        Dop,
        x0=np.zeros(m),
        tau=tau,
        mu=mu,
        niter=200,
        show=True,
    )[0]

    assert_array_almost_equal(xpd, xapd, decimal=2)
