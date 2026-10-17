import numpy as np
import pytest

from pyproximal import L21_plus_L1


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("sigma,rho", [(1.0, 1.0), (0.0, 0.8)])
def test_sparse_group_limits(dtype, axis, sigma, rho):
    """Pure L1 and zero regularization are finite, including zero groups."""
    x = np.array([[0, 1, 3], [0, 0.5, -4]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x *= (1 + 1j) / np.sqrt(2)
    tau = 0.5
    expected = np.array(
        [
            0 if abs(value) <= tau * sigma else value * (1 - tau * sigma / abs(value))
            for value in x.ravel()
        ]
    ).reshape(x.shape)
    with np.errstate(divide="raise", invalid="raise"):
        result = L21_plus_L1(sigma=sigma, rho=rho).prox(x, tau, axis=axis)
    np.testing.assert_allclose(result, expected, atol=1e-14)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_group_row_axis(dtype):
    """Grouping rows is equivalent to grouping columns of the transpose."""
    x = np.array([[0, 1, 3], [0, 0.5, -4]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x *= (1 + 1j) / np.sqrt(2)
    operator = L21_plus_L1(sigma=0.8, rho=0.4)
    expected = operator.prox(x.T, 0.5).T
    np.testing.assert_allclose(operator.prox(x, 0.5, axis=1), expected)
