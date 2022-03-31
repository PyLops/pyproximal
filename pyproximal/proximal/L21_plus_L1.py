import numpy as np
from pyproximal import ProxOperator
from pyproximal.ProxOperator import _check_tau


class L21_plus_L1(ProxOperator):
    r"""L21 + L1 norm proximal operator.

    Proximal operator of the :math:`L_{2,1} + L_1` mixed-norm:
    :math:`f(\mathbf{X}) = \sigma \rho \|\mathbf{X}\|_1 +
    \sigma (1 - \rho) \|\mathbf{X}\|_{2,1}`

    Parameters
    ----------
    sigma : :obj:`int`, optional
        Multiplicative coefficient of :math:`L_{2,1} + L_1` mixed-norm
    rho : :obj:`int`, optional
        Balancing between sparsity of :math:`L_1` and grouping of :math:`L_{2,1}`

    Notes
    -----
    The proximal operator of the :math:`L_{2,1} + L_1` mixed-norm is simply the
    product of each individual proximal operator [1]_.

    .. [1] Gramfort, Alexandre, Daniel Strohmeier, Jens Haueisen, Matti Hamalainen,
        and Matthieu Kowalski. "Functional brain imaging with M/EEG using structured
        sparsity in time-frequency dictionaries." In Biennial International Conference
        on Information Processing in Medical Imaging, pp. 600-611. Springer, Berlin,
        Heidelberg, 2011.
    """

    def __init__(self, sigma=1.0, rho=0.8):
        super().__init__(None, False)
        self.sigma = sigma
        self.rho = rho

    def __call__(self, x):
        return self.rho * self.sigma * np.sum(np.abs(x)) + \
               (1 - self.rho) * self.sigma * np.sum(np.sqrt(np.sum(x ** 2, axis=0)))

    @_check_tau
    def prox(self, x, tau, axis=0):
        thresh = self.sigma * tau
        l1 = np.maximum(np.abs(x) - thresh * self.rho, 0)
        # Axis defines what dimension to perform grouping over
        aux_l21 = np.sqrt(np.sum(np.maximum(
            np.abs(x) - thresh * self.rho, 0) ** 2, axis=axis))
        l21 = np.maximum(1 - thresh * (1 - self.rho) / aux_l21, 0)
        x = np.nan_to_num(x / np.abs(x)) * l1 * l21
        return x
