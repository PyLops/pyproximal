import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau


class L21_plus_L1(ProxOperator):
    r"""L21 + L1 norm proximal operator.

    Proximal operator of the :math:`L_{2,1} + L_1` mixed-norm:
    :math:`f(\mathbf{X}) = \sigma \rho \|\mathbf{X}\|_1 +
    \sigma (1 - \rho) \|\mathbf{X}\|_{2,1}`

    Parameters
    ----------
    sigma : :obj:`float`, optional
        Multiplicative coefficient of :math:`L_{2,1} + L_1` mixed-norm
    rho : :obj:`float`, optional
        Balancing between sparsity of :math:`L_1` and grouping of :math:`L_{2,1}`

    Notes
    -----
    The proximal operator of the :math:`L_{2,1} + L_1` mixed-norm is simply the
    product of each individual proximal operator [1]_.

    Grouping is performed along ``axis`` in :meth:`prox` (columns by default).
    When ``rho=1``, only elementwise L1 shrinkage remains. With ``sigma=0``,
    the proximal is the identity. Zero groups remain zero in both cases.

    .. [1] Gramfort, Alexandre, Daniel Strohmeier, Jens Haueisen, Matti Hamalainen,
        and Matthieu Kowalski. "Functional brain imaging with M/EEG using structured
        sparsity in time-frequency dictionaries." In Biennial International Conference
        on Information Processing in Medical Imaging, pp. 600-611. Springer, Berlin,
        Heidelberg, 2011.
    """

    def __init__(self, sigma: float = 1.0, rho: float = 0.8) -> None:
        super().__init__(None, False)
        self.sigma = sigma
        self.rho = rho

    def __call__(self, x: NDArray) -> float:
        return float(
            self.rho * self.sigma * np.sum(np.abs(x))
            + (1 - self.rho)
            * self.sigma
            * np.sum(np.sqrt(np.sum(np.abs(x) ** 2, axis=0)))
        )

    @_check_tau
    def prox(self, x: NDArray, tau: float, axis: int = 0) -> NDArray:
        thresh = self.sigma * tau
        absx = np.abs(x)
        l1 = np.maximum(absx - thresh * self.rho, 0)
        # Axis defines what dimension to perform grouping over
        aux_l21 = np.sqrt(np.sum(l1**2, axis=axis, keepdims=True))
        denominator = np.maximum(aux_l21, thresh * (1 - self.rho))
        denominator = np.where(denominator == 0, 1, denominator)
        l21 = np.maximum(1 - thresh * (1 - self.rho) / denominator, 0)
        phase = np.where(absx == 0, 0, x) / np.where(absx == 0, 1, absx)
        x = phase * l1 * l21
        return x
