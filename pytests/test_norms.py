import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pylops.basicoperators import (
    Diagonal,
    FirstDerivative,
    Gradient,
    Identity,
    MatrixMult,
)
from pylops.signalprocessing import Convolve1D

from pyproximal.proximal import (
    L0,
    L1,
    L2,
    L21,
    TV,
    Euclidean,
    Huber,
    HuberCircular,
    L2Convolve,
    L21_plus_L1,
    Nuclear,
    RelaxedMumfordShah,
)
from pyproximal.utils import moreau

par1 = {"nx": 10, "sigma": 1.0, "dtype": "float32"}  # even float32
par2 = {"nx": 11, "sigma": 2.0, "dtype": "float64"}  # odd float64

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Euclidean(par):
    """Euclidean norm and proximal/dual proximal"""
    np.random.seed(10)

    eucl = Euclidean(sigma=par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert eucl(x) == par["sigma"] * np.linalg.norm(x)

    # grad
    assert_array_almost_equal(eucl.grad(x), par["sigma"] * x / np.linalg.norm(x))

    # prox / dualprox (checking also verbosity)
    tau = 2.0
    assert moreau(eucl, x, tau, verb=True)


def test_L2_solver():
    """Check L2 raises an error if the solver is not recognized"""
    with pytest.raises(ValueError, match="Available options are:"):
        d = np.random.normal(0.0, 1.0, 10)
        _ = L2(Op=Diagonal(d), solver="foo")


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("niter", [500, lambda x: 500])
def test_L2_op(par, niter):
    """L2 norm of Op*x and grad and proximal"""
    np.random.seed(10)

    b = np.zeros(par["nx"], dtype=par["dtype"])
    d = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    l2 = L2(Op=Diagonal(d, dtype=par["dtype"]), b=b, sigma=par["sigma"], niter=niter)

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l2(x) == (par["sigma"] / 2.0) * np.linalg.norm(d * x) ** 2

    # grad: since Op is a Diagonal operator the gradient becomes
    # -sigma*d[i]*(b[i] - d[i]*x[i]) for every i
    assert_array_almost_equal(l2.grad(x), -par["sigma"] * d * (b - d * x))

    # prox: since Op is a Diagonal operator the denominator becomes
    # 1 + sigma*tau*d[i] for every i
    tau = 2.0
    den = 1.0 + par["sigma"] * tau * d**2
    assert_array_almost_equal(l2.prox(x, tau), x / den, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("explicit", [False, True])
def test_L2_op_solver(par, explicit):
    """L2 norm of Op*x-b and proximal, the first compared to close-form
    solution and the second with different choices of solver."""
    np.random.seed(10)

    for q in (None, np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])):
        Op = MatrixMult(
            np.random.normal(0, 1, (par["nx"], par["nx"])).astype(dtype=par["dtype"]),
            dtype=par["dtype"],
        )
        Op.explicit = explicit
        b = np.ones(par["nx"], dtype=par["dtype"])
        l2_leg = L2(
            Op=Op, b=b, sigma=par["sigma"], q=q, solver="legacy", niter=2 * par["nx"]
        )
        l2_cg = L2(
            Op=Op, b=b, sigma=par["sigma"], q=q, solver="cg", niter=2 * par["nx"]
        )
        l2_cgls = L2(
            Op=Op, b=b, sigma=par["sigma"], q=q, solver="cgls", niter=2 * par["nx"]
        )

        # norm
        x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
        norm = (par["sigma"] / 2.0) * np.linalg.norm(Op * x - b) ** 2
        if q is not None:
            norm += np.dot(q, x)
        assert l2_leg(x) == norm

        # prox
        tau = 2.0
        prox_leg = l2_leg.prox(x, tau)
        prox_cg = l2_cg.prox(x, tau)
        prox_cgls = l2_cgls.prox(x, tau)

        assert_array_almost_equal(prox_leg, prox_cg, decimal=4)
        assert_array_almost_equal(prox_leg, prox_cgls, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2_dense(par):
    """L2 norm of Op*x with dense Op and proximal
    compared to closed-form solution (since Op is a Diagonal
    operator the denominator becomes 1 + sigma*tau*d[i]^2 for
    every i)"""
    np.random.seed(10)

    for densesolver in ("numpy", "scipy", "factorize"):
        b = np.zeros(par["nx"], dtype=par["dtype"])
        d = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
        l2 = L2(
            Op=MatrixMult(np.diag(d), dtype=par["dtype"]),
            b=b,
            sigma=par["sigma"],
            densesolver=densesolver,
        )

        # norm
        x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
        assert l2(x) == (par["sigma"] / 2.0) * np.linalg.norm(d * x) ** 2

        # prox
        tau = 2.0
        den = 1.0 + par["sigma"] * tau * d**2
        assert_array_almost_equal(l2.prox(x, tau), x / den, decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2_diff(par):
    """L2 norm of difference (x-b) and proximal
    compared to closed-form solution"""
    np.random.seed(10)

    b = np.ones(par["nx"], dtype=par["dtype"])
    l2 = L2(b=b, sigma=par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l2(x) == (par["sigma"] / 2.0) * np.linalg.norm(x - b) ** 2

    # prox
    tau = 2.0
    assert_array_almost_equal(
        l2.prox(x, tau), (x + par["sigma"] * tau * b) / (1.0 + par["sigma"] * tau)
    )


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2_x(par):
    """L2 norm of x and grad and proximal (implemented both directly and
    with identity operator and zero b and compared to closed-form
    solution)"""
    np.random.seed(10)

    for q in (None, np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])):
        l2 = L2(
            Op=Identity(par["nx"], dtype=par["dtype"]),
            b=np.zeros(par["nx"], dtype=par["dtype"]),
            sigma=par["sigma"],
            q=q,
        )
        l2direct = L2(
            sigma=par["sigma"],
            q=q,
        )

        # norm
        x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
        norm = (par["sigma"] / 2.0) * np.linalg.norm(x) ** 2
        if q is not None:
            norm += np.dot(q, x)
        assert l2(x) == norm
        assert l2direct(x) == norm

        # grad
        grad = par["sigma"] * x
        if q is not None:
            grad += q
        assert_array_almost_equal(l2.grad(x), grad)

        # prox
        tau = 2.0
        if q is not None:
            prox = (x - tau * q) / (1.0 + par["sigma"] * tau)
        else:
            prox = x / (1.0 + par["sigma"] * tau)
        assert_array_almost_equal(l2.prox(x, tau), prox)
        assert_array_almost_equal(l2direct.prox(x, tau), prox)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L2Convolve(par):
    """L2Convolve norm of x and grad and prox/dualprox (norm and grad are compared
    to solution in the time domain)"""
    np.random.seed(10)

    # 1d
    h = np.ones(3) / 3.0
    Cop = Convolve1D(par["nx"], h=h, offset=1, dtype=par["dtype"])
    b = np.zeros(par["nx"], dtype=par["dtype"])

    l2 = L2Convolve(
        h=h,
        b=b,
        sigma=par["sigma"],
    )

    # define x with a spike in the middle to
    # avoid boundary effects in the convolution
    # in the time vs frequency domain
    x = np.zeros(par["nx"], dtype=par["dtype"])
    x[par["nx"] // 2] = 1.0

    # norm
    norm = (par["sigma"] / 2.0) * np.linalg.norm(b - Cop @ x) ** 2
    assert l2(x) == pytest.approx(norm)

    # grad
    grad = par["sigma"] * Cop.H @ (Cop @ x - b)
    assert_array_almost_equal(l2.grad(x), grad)

    # prox / dualprox
    tau = 2.0
    moreau(l2, x, tau)

    # 2d on first
    h = np.ones(3) / 3.0
    Cop = Convolve1D((par["nx"], 4), h=h, offset=1, axis=0, dtype=par["dtype"])
    b = np.zeros((par["nx"], 4), dtype=par["dtype"])

    l2 = L2Convolve(
        h=h,
        b=b,
        sigma=par["sigma"],
        dims=(par["nx"], 4),
        dir=0,
    )

    # define x with a spike in the middle to
    # avoid boundary effects in the convolution
    # in the time vs frequency domain
    x = np.zeros((par["nx"], 4), dtype=par["dtype"])
    x[par["nx"] // 2] = 1.0

    # norm
    norm = (par["sigma"] / 2.0) * np.linalg.norm(b - Cop @ x) ** 2
    assert l2(x) == pytest.approx(norm)

    # grad
    grad = par["sigma"] * Cop.H @ (Cop @ x - b)
    assert_array_almost_equal(l2.grad(x).reshape(par["nx"], 4), grad)

    # prox / dualprox
    tau = 2.0
    moreau(l2, x.ravel(), tau)

    # 2d on last
    h = np.ones(3) / 3.0
    Cop = Convolve1D((4, par["nx"]), h=h, offset=1, axis=-1, dtype=par["dtype"])
    b = np.zeros((4, par["nx"]), dtype=par["dtype"])

    l2 = L2Convolve(
        h=h,
        b=b,
        sigma=par["sigma"],
        dims=(4, par["nx"]),
        dir=-1,
    )

    # define x with a spike in the middle to
    # avoid boundary effects in the convolution
    # in the time vs frequency domain
    x = np.zeros((4, par["nx"]), dtype=par["dtype"])
    x[:, par["nx"] // 2] = 1.0

    # norm
    norm = (par["sigma"] / 2.0) * np.linalg.norm(b - Cop @ x) ** 2
    assert l2(x) == pytest.approx(norm)

    # grad
    grad = par["sigma"] * Cop.H @ (Cop @ x - b)
    assert_array_almost_equal(l2.grad(x).reshape(4, par["nx"]), grad)

    # prox / dualprox
    tau = 2.0
    moreau(l2, x.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1(par):
    """L1 norm and grad and proximal/dual proximal"""
    np.random.seed(10)

    l1 = L1(sigma=par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l1(x) == par["sigma"] * np.sum(np.abs(x))

    # grad (note that since this norm is non-smooth, the
    # gradient method is not implemented; as such the gradient
    # of the Moreau envelope is called instead
    _ = l1.grad(x)

    # prox / dualprox
    tau = 2.0
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1_func(par):
    """L1 norm and proximal/dual proximal with sigma as callable"""
    np.random.seed(10)

    l1 = L1(sigma=lambda x: par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l1(x) == par["sigma"] * np.sum(np.abs(x))

    # prox / dualprox
    tau = 2.0
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L1_diff(par):
    """L1 norm of difference and proximal/dual proximal"""
    np.random.seed(10)

    g = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    l1 = L1(sigma=par["sigma"], g=g)

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l1(x) == par["sigma"] * np.sum(np.abs(x - g))

    # prox / dualprox
    tau = 2.0
    assert moreau(l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L0(par):
    """L0 norm and proximal/dual proximal"""
    np.random.seed(10)

    l0 = L0(sigma=par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert l0(x) == np.sum(np.abs(x) > 0.0)

    # prox / dualprox
    tau = 2.0
    assert moreau(l0, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L21(par):
    """L21 norm and proximal/dual proximal on 2d array (compare it with N
    L2 norms on the columns of the 2d array
    """
    np.random.seed(10)

    l21 = L21(ndim=2, sigma=par["sigma"])

    # norm
    x = np.random.normal(0.0, 1.0, 2 * par["nx"]).astype(par["dtype"])
    l21_ = par["sigma"] * np.sum(np.linalg.norm(x.reshape(2, par["nx"]), axis=0))
    assert_array_almost_equal(l21(x), l21_)

    # prox / dualprox
    tau = 2.0
    assert moreau(l21, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_L21_plus_L1(par):
    """L21 plus L1 norm on 2darray."""
    np.random.seed(10)

    l21_plus_l1 = L21_plus_L1(sigma=par["sigma"], rho=0.8)

    # norm
    x = np.random.normal(0.0, 1.0, (2 * par["nx"], par["nx"])).astype(par["dtype"])
    rho = 0.8
    l21_plus_l1_ = par["sigma"] * rho * np.sum(np.abs(x)) + par["sigma"] * (
        1 - rho
    ) * np.sum(np.sqrt(np.sum(x**2, axis=0)))
    assert_array_almost_equal(l21_plus_l1(x), l21_plus_l1_)

    tau = 2.0
    l21_plus_l1.prox(x, tau)

    assert moreau(l21_plus_l1, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Huber(par):
    """Huber norm and proximal/dual proximal"""
    np.random.seed(10)

    hub = Huber(alpha=par["sigma"])

    # norm
    x = np.random.uniform(0.0, 0.1 * par["sigma"], par["nx"]).astype(par["dtype"])
    assert hub(x) == np.sum(x**2) / (2 * par["sigma"])

    # prox / dualprox
    tau = 2.0
    assert moreau(hub, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_HuberCircular(par):
    """Circular Huber norm and proximal/dual proximal"""
    np.random.seed(10)

    hub = HuberCircular(alpha=par["sigma"])

    # norm
    x = np.random.uniform(0.0, 0.1, par["nx"]).astype(par["dtype"])
    x_below = (
        (0.1 * par["sigma"]) * x / np.linalg.norm(x)
    )  # to ensure that is smaller than sigma
    assert hub(x_below) == np.linalg.norm(x_below) ** 2 / (2 * par["sigma"])

    x_above = (
        (2.0 * par["sigma"]) * x / np.linalg.norm(x)
    )  # to ensure that is larger than sigma
    assert hub(x_above) == np.linalg.norm(x_above) - 0.5 * par["sigma"]

    # prox / dualprox
    tau = 2.0
    assert moreau(hub, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_TV(par):
    """TV norm of x and proximal/dual proximal"""
    np.random.seed(10)

    # 1d
    tv = TV(dims=(par["nx"],), sigma=par["sigma"])
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])

    # norm
    derivOp = FirstDerivative(par["nx"], dtype=par["dtype"], kind="forward")
    dx = derivOp @ x
    assert_array_almost_equal(tv(x), par["sigma"] * np.sum(np.abs(dx), axis=0))

    # prox / dualprox
    tau = 2.0
    assert moreau(tv, x, tau)

    # 2d/3d/4d
    dims = ((4, par["nx"]), (4, 8, par["nx"]), (4, 4, 8, par["nx"]))

    for dim in dims:
        tv = TV(dims=dim, sigma=par["sigma"])
        x = np.random.normal(0.0, 1.0, dim).astype(par["dtype"])

        # norm
        derivOp = Gradient(dim, dtype=par["dtype"], kind="forward")
        dx = derivOp @ x
        f = 0.0
        for g in dx:
            f += np.power(abs(g), 2)
        assert_array_almost_equal(tv(x), par["sigma"] * np.sum(np.sqrt(f)))

        # prox / dualprox
        tau = 2.0
        assert moreau(tv, x.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
@pytest.mark.parametrize("kappa", [1.0, lambda x: 1.0])
def test_rMS(par, kappa):
    """rMS norm and proximal/dual proximal"""
    rMS = RelaxedMumfordShah(sigma=par["sigma"], kappa=kappa)

    # norm
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert rMS(x) == np.minimum(par["sigma"] * np.linalg.norm(x) ** 2, 1.0)

    # prox / dualprox
    tau = 2.0
    assert moreau(rMS, x, tau)


def test_Nuclear_FOM():
    """Nuclear norm benchmark with FOM solver"""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    proxX = np.array([[1.4089, 1.8613, 2.3137], [3.3640, 4.4441, 5.5242]])
    nucl = Nuclear(X.shape)
    assert_array_almost_equal(proxX.ravel(), nucl.prox(X, 1), decimal=4)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Nuclear(par):
    """Nuclear norm and proximal/dual proximal"""
    np.random.seed(10)

    nucl = Nuclear((par["nx"], 2 * par["nx"]), sigma=par["sigma"])

    # norm, cross-check with svd (use tolerance as two methods don't provide
    # the exact same eigenvalues)
    X = np.random.uniform(0.0, 0.1, (par["nx"], 2 * par["nx"])).astype(par["dtype"])
    _, S, _ = np.linalg.svd(X)
    assert (nucl(X.ravel()) - par["sigma"] * np.sum(S)) < 1e-3

    # prox / dualprox
    tau = 2.0
    assert moreau(nucl, X.ravel(), tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Weighted_Nuclear(par):
    """Weighted nuclear norm and proximal/dual proximal"""
    np.random.seed(10)

    weights = par["sigma"] * np.linspace(0.1, 5, 2 * par["nx"])
    nucl = Nuclear((par["nx"], 2 * par["nx"]), sigma=weights)

    # norm, cross-check with svd (use tolerance as two methods don't provide
    # the exact same singular values)
    X = np.random.uniform(0.0, 0.1, (par["nx"], 2 * par["nx"])).astype(par["dtype"])
    S = np.linalg.svd(X, compute_uv=False)
    assert (nucl(X.ravel()) - np.sum(weights[: S.size] * S)) < 1e-2

    # prox / dualprox
    tau = 2.0
    assert moreau(nucl, X.ravel(), tau)
