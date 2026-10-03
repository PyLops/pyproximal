"""Skeleton for a new PyProximal proximal operator.

Copy into ``pyproximal/proximal/<MyProx>.py`` (UpperCamelCase, matching the class),
rename, and fill in. The bodies below implement a scaled L1 norm only so that the
skeleton runs and passes ``moreau`` as-is: replace them with the maths of the new
operator. Delete anything that does not apply.
"""

import numpy as np
from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau


class MyProx(ProxOperator):
    r"""One-line summary of the proximal operator.

    Proximal operator of the function
    :math:`f(\mathbf{x}) = \sigma \|\mathbf{x}\|_1` (replace with the
    definition of the new function).

    .. versionadded:: X.Y.Z

    Parameters
    ----------
    sigma : :obj:`float`, optional
        Multiplicative coefficient of the function.

    Raises
    ------
    ValueError
        If ``sigma`` is not positive.

    Notes
    -----
    The proximal operator of :math:`f` is defined as [1]_:

    .. math::

        \prox_{\tau f}(\mathbf{x}) = \ldots

    and, when a closed form is available, its dual proximal operator as:

    .. math::

        \prox_{\tau f^*}(\mathbf{x}) = \ldots

    .. [1] Author, A., and Author, B., "Title of the paper", Journal,
        volume, pp. xx-yy. Year.

    """

    def __init__(self, sigma: float = 1.0) -> None:
        # Op: PyLops linear operator used inside the function (None if not needed)
        # hasgrad: True only if f is differentiable and ``grad`` is overridden
        super().__init__(None, False)
        if sigma <= 0:
            msg = "sigma must be positive"
            raise ValueError(msg)
        self.sigma = sigma

    def __call__(self, x: NDArray) -> float:
        # function value (return ``bool`` membership for indicator functions)
        return float(self.sigma * np.sum(np.abs(x)))

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        # use ``ncp`` (not ``np``) for array operations so that CuPy inputs work
        ncp = get_array_module(x)
        thresh = tau * self.sigma
        return ncp.maximum(ncp.abs(x) - thresh, 0.0) * ncp.sign(x)

    @_check_tau
    def proxdual(self, x: NDArray, tau: float) -> NDArray:
        # implement only if a closed form exists; otherwise delete this method
        # and the base class derives it from ``prox`` via the Moreau identity
        ncp = get_array_module(x)
        return ncp.clip(x, -self.sigma, self.sigma)

    # def grad(self, x: NDArray) -> NDArray:
    #     # only with ``hasgrad=True``: true gradient of f (otherwise the base
    #     # class returns the gradient of the Moreau envelope)
    #     return ...
