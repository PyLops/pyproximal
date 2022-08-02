import pytest

import numpy as np

from pyproximal.utils import moreau
from pyproximal.proximal import ETP, Geman, Log, Log1, SCAD, QuadraticEnvelopeCard

par1 = {'nx': 10, 'sigma': 1., 'a': 2.1, 'gamma': 0.5, 'mu': 0.5, 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'sigma': 2., 'a': 3.7, 'gamma': 5.0, 'mu': 1.5, 'dtype': 'float64'}  # odd float64


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_SCAD(par):
    """SCAD penalty and proximal/dual proximal
    """
    np.random.seed(10)
    scad = SCAD(sigma=par['sigma'], a=par['a'])
    # SCAD should behave like l1-norm when values are small
    x = np.random.normal(0., 0.1, par['nx']).astype(par['dtype'])
    assert scad(x) == par['sigma'] * np.linalg.norm(x, 1)

    # SCAD should behave like hard-thresholding when values are large
    x = np.random.normal(10., 0.1, par['nx']).astype(par['dtype'])
    assert scad(x) == pytest.approx((par['a'] + 1) * par['sigma'] ** 2 / 2 * np.count_nonzero(x))

    # Check proximal operator
    tau = 2.
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    assert moreau(scad, x, tau)

    # Make sure that a ValueError is raised if you try to instantiate with a <= 2
    with pytest.raises(ValueError):
        _ = SCAD(sigma=1.0, a=1.7)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Log(par):
    """Log penalty and proximal/dual proximal
    """
    np.random.seed(10)
    log = Log(sigma=par['sigma'], gamma=par['gamma'])
    # Log
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    expected = par['sigma'] / np.log(par['gamma'] + 1) * np.linalg.norm(np.log(par['gamma'] * np.abs(x) + 1), 1)
    assert log(x) == pytest.approx(expected)

    # Check proximal operator
    tau = 2.
    assert moreau(log, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Log1(par):
    """Log1 penalty and proximal/dual proximal
    """
    np.random.seed(10)
    log = Log1(sigma=par['sigma'])

    # Check proximal operator
    tau = 2.
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    assert moreau(log, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_ETP(par):
    """Exponential-type penalty and proximal/dual proximal
    """
    np.random.seed(10)
    etp = ETP(sigma=par['sigma'], gamma=par['gamma'])
    # ETP
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    expected = par['sigma'] / (1 - np.exp(-par['gamma'])) * np.linalg.norm((1 - np.exp(-par['gamma'] * np.abs(x))), 1)
    assert etp(x) == pytest.approx(expected)

    # Check proximal operator
    tau = 2.
    assert moreau(etp, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_Geman(par):
    """Geman penalty and proximal/dual proximal
    """
    np.random.seed(10)
    geman = Geman(sigma=par['sigma'], gamma=par['gamma'])
    # Geman
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    expected = par['sigma'] * np.linalg.norm(np.abs(x) / (np.abs(x) + par['gamma']), 1)
    assert geman(x) == pytest.approx(expected)

    # Check proximal operator
    tau = 2.
    assert moreau(geman, x, tau)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_QuadraticEnvelopeCard(par):
    """QuadraticEnvelopeCard penalty and proximal/dual proximal
    """
    np.random.seed(10)
    fmu = QuadraticEnvelopeCard(mu=par['mu'])
    # Quadratic envelope of the l0-penalty
    x = np.random.normal(0., 10.0, par['nx']).astype(par['dtype'])
    expected = np.linalg.norm(par['mu'] - 0.5 * np.maximum(0, np.sqrt(2 * par['mu']) - np.abs(x)) ** 2, 1)
    assert fmu(x) == pytest.approx(expected)

    # Check proximal operator
    tau = 0.25
    assert moreau(fmu, x, tau)
