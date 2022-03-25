import pytest

import numpy as np

from pyproximal.utils import moreau
from pyproximal.proximal import SCAD

par1 = {'nx': 10, 'sigma': 1., 'a': 2.1, 'dtype': 'float32'}  # even float32
par2 = {'nx': 11, 'sigma': 2., 'a': 3.7, 'dtype': 'float64'}  # odd float64

np.random.seed(10)


@pytest.mark.parametrize("par", [(par1), (par2)])
def test_SCAD(par):
    """SCAD penalty and proximal/dual proximal
    """
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