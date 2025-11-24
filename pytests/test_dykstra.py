from typing import Any, Callable, Dict

import numpy as np
import pytest

from pyproximal.projection import (
    BoxProj,
    EuclideanBallProj,
    GenericIntersectionProj,
    HalfSpaceProj,
)
from pyproximal.proximal import (
    L1,
    L2,
    L21,
    Box,
    GenericIntersectionProx,
    L21_plus_L1,
    Sum,
)
from pyproximal.utils import moreau

par1proj = {"nx": 10, "ny": 8, "axis": 0, "dtype": "float32"}  # even float32 dir0
par2proj = {"nx": 11, "ny": 8, "axis": 1, "dtype": "float64"}  # odd float64 dir1

par1prox = {"nx": 10, "ny": 10, "sigma": 1.0, "dtype": "float32"}  # even float32
par2prox = {"nx": 11, "ny": 14, "sigma": 2.0, "dtype": "float64"}  # odd float64


@pytest.mark.parametrize("par", [(par1proj), (par2proj)])
def test_GenericIntersectionProx(par: Dict[str, Any]) -> None:
    """GenericIntersectionProx and proximal/dual proximal of related indicator"""
    rng = np.random.default_rng(10)

    w = rng.normal(10.0, 1.0, par["nx"]).astype(par["dtype"])
    w /= np.linalg.norm(w)  # unit normal toward (10, ..., 10)
    b = rng.uniform(0.1, 0.45)
    half_space = HalfSpaceProj(w, b)

    # radius is larger than b of the halfspace
    eucl = EuclideanBallProj(np.zeros(par["nx"]), rng.uniform(0.5, 1.0))

    # box is not smaller than the Euclidean ball
    box = BoxProj(rng.uniform(-1.0, -0.5, par["nx"]), rng.uniform(0.5, 1.0, par["nx"]))

    # init x around (10, ..., 10) so that x is outside the halfspace
    x = rng.normal(10.0, 1.0, par["nx"]).astype(par["dtype"])

    tau = 2.0

    projections: list[list[Callable[[Any], Any]]] = [
        # single projection
        [eucl],
        [box],
        [half_space],
        # two projections
        [eucl, box],
        [box, eucl],
        [half_space, eucl],
        [eucl, half_space],
        [box, half_space],
        [half_space, box],
        [eucl, eucl],
        [box, box],
        [half_space, half_space],
        # three projections
        [half_space, eucl, box],
        [half_space, box, eucl],
        [eucl, half_space, box],
        [box, half_space, eucl],
        [eucl, box, half_space],
        [box, eucl, half_space],
    ]
    for proj in projections:

        # different torelance for float32 and float64
        tol = (
            float(np.finfo(par["dtype"]).resolution) * 10.0
        )  # pylint: disable=no-member

        d = GenericIntersectionProx(proj, tol=tol, niter=1000)

        # evaluation
        assert d(x) is False
        xp = d.prox(x, 1.0)
        assert d(xp) is True

        # prox / dualprox
        tau = 2.0
        assert moreau(d, x, tau)

        if len(proj) == 2:
            d = GenericIntersectionProx(proj, tol=tol, niter=1000, use_parallel=True)

            # evaluation
            assert d(x) is False
            xp = d.prox(x, 1.0)
            assert d(xp) is True

            # prox / dualprox
            tau = 2.0
            assert moreau(d, x, tau)


@pytest.mark.parametrize("par", [(par1prox), (par2prox)])
def test_Sum_l1_l1(par: Dict[str, Any]) -> None:
    """Check Sum for L1 + L1"""

    atol = 1e-6
    tau = 1.0
    rng = np.random.default_rng(10)

    x = rng.normal(0.0, 3.5, par["nx"]).astype(par["dtype"])
    sigma_1 = rng.uniform(0.1, 1.0)
    sigma_2 = rng.uniform(0.1, 1.0)

    l1_1 = L1(sigma=sigma_1)
    l1_2 = L1(sigma=sigma_2)
    l1_l1 = L1(sigma=sigma_1 + sigma_2)

    d = Sum([l1_1, l1_2])
    assert np.allclose(l1_l1(x), d(x), atol=atol)
    assert np.allclose(l1_l1.prox(x, tau), d.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    d = Sum([l1_2, l1_1])
    assert np.allclose(l1_l1(x), d(x), atol=atol)
    assert np.allclose(l1_l1.prox(x, tau), d.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    weights = rng.uniform(0.25, 0.75, 2)
    weights /= weights.sum()
    dp = Sum([l1_1, l1_2], use_parallel=True, weights=weights)
    assert np.allclose(l1_l1(x), d(x), atol=atol)
    assert np.allclose(l1_l1.prox(x, tau), dp.prox(x, tau), atol=atol)
    assert moreau(dp, x, tau)


@pytest.mark.parametrize("par", [(par1prox), (par2prox)])
def test_Sum_l2_l2(par: Dict[str, Any]) -> None:
    """Check Sum for L2 + L2"""

    atol = 1e-6
    tau = 1.0
    rng = np.random.default_rng(10)

    x = rng.normal(0.0, 3.5, par["nx"]).astype(par["dtype"])
    sigma_1 = rng.uniform(0.1, 1.0)
    sigma_2 = rng.uniform(0.1, 1.0)

    l2_1 = L2(sigma=sigma_1)
    l2_2 = L2(sigma=sigma_2)
    l2_l2 = L2(sigma=sigma_1 + sigma_2)

    d = Sum([l2_1, l2_2])
    assert np.allclose(l2_l2(x), d(x), atol=atol)
    assert np.allclose(l2_l2.prox(x, tau), d.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    d = Sum([l2_2, l2_1])
    assert np.allclose(l2_l2(x), d(x), atol=atol)
    assert np.allclose(l2_l2.prox(x, tau), d.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    weights = rng.uniform(0.25, 0.75, 2)
    weights /= weights.sum()
    dp = Sum([l2_1, l2_2], use_parallel=True, weights=weights)
    assert np.allclose(l2_l2(x), d(x), atol=atol)
    assert np.allclose(l2_l2.prox(x, tau), dp.prox(x, tau), atol=atol)
    assert moreau(dp, x, tau)


@pytest.mark.parametrize("par", [(par1prox), (par2prox)])
def test_Sum_l21_l1(par: Dict[str, Any]) -> None:
    """Check Sum for L21 + L1"""

    atol = 1e-5
    tau = 1.0
    rng = np.random.default_rng(10)

    x = rng.normal(0.0, 3.5, par["nx"]).astype(par["dtype"])
    rho = rng.uniform(0.0, 1.0)
    sigma = rng.uniform(0.1, 4.0)
    l21_l1 = L21_plus_L1(sigma=sigma, rho=rho)
    l1 = L1(sigma=sigma * rho)
    l21 = L21(sigma=sigma * (1 - rho), ndim=par["nx"])

    d = Sum([l1, l21])
    assert np.allclose(d(x), l21_l1(x), atol=atol)
    assert np.allclose(d.prox(x, tau), l21_l1.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    d = Sum([l21, l1])
    assert np.allclose(d(x), l21_l1(x), atol=atol)
    assert np.allclose(d.prox(x, tau), l21_l1.prox(x, tau), atol=atol)
    assert moreau(d, x, tau)

    weights = rng.uniform(0.1, 1.0, 2)
    weights /= weights.sum()
    dp = Sum([l21, l1], use_parallel=True, weights=weights)
    assert np.allclose(d(x), l21_l1(x), atol=atol)
    assert np.allclose(dp.prox(x, tau), l21_l1.prox(x, tau), atol=atol)
    assert moreau(dp, x, tau)


@pytest.mark.parametrize("par", [(par1prox), (par2prox)])
def test_Sum_f1f2f3f4(par: Dict[str, Any]) -> None:
    """Check Sum for f1+f2+f3+f4"""

    tau = 1.0
    rng = np.random.default_rng(10)

    x = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])
    g = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])
    b = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])
    q = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])

    l1 = L1(sigma=rng.uniform(0.1, 1.0), g=g)
    l2 = L2(sigma=rng.uniform(0.1, 1.0), b=b, q=q, alpha=rng.uniform(0.1, 1.0))
    l21_l1 = L21_plus_L1(sigma=rng.uniform(0.1, 1.0), rho=rng.uniform(0.1, 1.0))
    l21 = L21(sigma=rng.uniform(0.1, 1.0), ndim=par["nx"])

    for prox_ops in [
        [l1, l2],
        [l2, l1, l2],
        [l1, l2, l1, l21],
        [l1, l2, l21, l21_l1],
    ]:
        d = Sum(prox_ops)
        assert moreau(d, x, tau)


@pytest.mark.parametrize("par", [(par1prox), (par2prox)])
def test_Sum_numeric_projection(par: Dict[str, Any]) -> None:
    """Check Sum for numeric prox + indicator function"""

    rng = np.random.default_rng(10)
    tau = 1.0

    g = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])
    l1 = L1(sigma=rng.uniform(0.1, 1.0), g=g)  # numeric prox

    box1 = Box(
        rng.uniform(-1.0, -0.1, par["nx"]), rng.uniform(0.1, 1.0, par["nx"])
    )  # boolean prox
    box2 = Box(
        rng.uniform(-1.0, -0.1, par["nx"]), rng.uniform(0.1, 1.0, par["nx"])
    )  # boolean prox

    # check: Moreau identity
    x = rng.normal(0.0, 2.5, par["nx"]).astype(par["dtype"])
    d = Sum([l1, box1])
    assert moreau(d, x, tau)

    # check: return numeric
    x_proj = box1.prox(x, tau)
    assert box1(x_proj)
    x_prox = d(x_proj)  # return numeric because x_proj is in the box
    assert isinstance(x_prox, float)

    # check: return False
    x_proj = box1.prox(x, tau)
    assert box1(x_proj)  # x_proj is in the box
    x_proj += rng.uniform(0.1, 1.0, par["nx"])
    assert not box1(x_proj)  # now x_proj is out of the box
    x_prox = d(x_proj)  # return False
    assert isinstance(x_prox, bool) and not x_prox

    # check: return True
    d_proj = GenericIntersectionProj(
        [
            lambda x: box1.prox(x, tau),
            lambda x: box2.prox(x, tau),
        ]
    )
    x_proj = d_proj(x)  # x_proj is in the intersection
    d = Sum([box1, box2])
    x_prox = d(x_proj)  # return True
    assert isinstance(x_prox, bool) and x_prox
    assert moreau(d, x, tau)
