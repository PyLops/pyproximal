import numpy as np
import pytest

from pyproximal import L1
from pyproximal.utils import moreau


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("shifted", [False, True])
def test_l1_complex_dual_projection(dtype, weighted, shifted):
    """Complex dual projection preserves phase and satisfies Moreau's identity."""
    x = np.array([0, 2j, 3 + 4j, -4 + 3j, 0.1 - 0.1j], dtype=dtype)
    sigma = np.array([0, 0.2, 0.7, 2, 0.1]) if weighted else 0.7
    g = np.array([0, 0.1j, 0.2, -0.3j, 0.1], dtype=dtype) if shifted else None
    tau = 0.3
    z = x if g is None else x - tau * g
    expected = np.array(
        [
            value if abs(value) <= radius else radius * value / abs(value)
            for value, radius in zip(z, np.broadcast_to(sigma, x.shape), strict=True)
        ]
    )
    operator = L1(sigma=sigma, g=g)
    np.testing.assert_allclose(operator.proxdual(x, tau), expected, atol=2e-7)
    assert moreau(operator, x, tau, tol=2e-6)


@pytest.mark.parametrize("complex_input", [False, True])
def test_l1_dual_current_weights(complex_input):
    """Dual bounds use the same callable weight as the latest primal call."""
    x = np.array([3.0, -4.0])
    if complex_input:
        x = x.astype(complex) * 1j
    operator = L1(sigma=lambda count: 1.0 + count)
    operator.prox(x, 0.5)
    np.testing.assert_allclose(operator.proxdual(x, 0.5), 2 * x / np.abs(x))
    assert operator.count == 1
    assert moreau(operator, x, 0.5)
