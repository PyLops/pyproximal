"""Skeleton tests for a new PyProximal proximal operator.

Merge into the ``pytests/test_*.py`` file that matches the operator
(``test_norms.py``, ``test_proximal.py``, ``test_concave_penalties.py``,
``test_projection.py``) and reuse its existing ``par*`` dicts. Keep the Moreau
test when both ``prox`` and ``proxdual`` have closed forms; otherwise rely on
the edge-case and brute-force tests.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.optimize import minimize

from pyproximal.proximal import MyProx  # adjust import
from pyproximal.utils import moreau

par1 = {"nx": 10, "sigma": 1.0, "dtype": "float32"}  # even float32
par2 = {"nx": 11, "sigma": 2.0, "dtype": "float64"}  # odd float64


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MyProx(par):
    """MyProx function and proximal/dual proximal"""
    np.random.seed(10)
    myprox = MyProx(sigma=par["sigma"])

    # function value, checked against an independent computation
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert myprox(x) == pytest.approx(par["sigma"] * np.sum(np.abs(x)), rel=1e-5)

    # prox / dualprox (only when both have a closed form)
    tau = 2.0
    assert moreau(myprox, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_MyProx_edgecases(par):
    """MyProx proximal on cases with a known answer"""
    np.random.seed(10)
    myprox = MyProx(sigma=par["sigma"])

    # prox of zero (replace with the known answer for the new function)
    x = np.zeros(par["nx"], dtype=par["dtype"])
    assert_array_almost_equal(myprox.prox(x, 1.0), x)

    # tau -> 0 leaves x unchanged
    x = np.random.normal(0.0, 1.0, par["nx"]).astype(par["dtype"])
    assert_array_almost_equal(myprox.prox(x, 1e-10), x, decimal=5)

    # brute-force minimization of f(y) + ||y - x||^2 / (2 tau) on a small vector
    tau = 0.5
    x = np.random.normal(0.0, 1.0, 5)
    y = minimize(
        lambda y: myprox(y) + np.sum((y - x) ** 2) / (2 * tau),
        x,
        method="Powell",
        options={"xtol": 1e-8, "ftol": 1e-10},
    ).x
    assert_array_almost_equal(myprox.prox(x, tau), y, decimal=4)


def test_MyProx_raises():
    """Check input validation of MyProx"""
    with pytest.raises(ValueError, match="sigma must be positive"):
        _ = MyProx(sigma=-1.0)
