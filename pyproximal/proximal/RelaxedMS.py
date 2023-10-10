import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.proximal.L1 import _current_sigma


def _l2(x, alpha):
    r"""Scaling operation.

    Applies the proximal of ``alpha||y - x||_2^2`` which is essentially a scaling operation.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Vector
    alpha : :obj:`float`
        Scaling parameter

    Returns
    -------
    y : :obj:`numpy.ndarray`
        Scaled vector

    """
    y = 1 / (1 + 2 * alpha) * x
    return y


def _current_kappa(kappa, count):
    if not callable(kappa):
        return kappa
    else:
        return kappa(count)


class RelaxedMumfordShah(ProxOperator):
    r"""Relaxed Mumford-Shah norm proximal operator.

    Proximal operator of the relaxed Mumford-Shah norm:
    :math:`\text{rMS}(x) = \min (\alpha\Vert x\Vert_2^2, \kappa)`.

    Parameters
    ----------
    sigma : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Multiplicative coefficient of L2 norm that controls the smoothness of the solutuon.
        This can be a constant number, a list of values (for multidimensional inputs, acting
        on the second dimension) or a function that is called passing a counter which keeps
        track of how many times the ``prox`` method has been invoked before and returns a
        scalar (or a list of) ``sigma`` to be used.
    kappa : :obj:`float` or :obj:`list` or :obj:`np.ndarray` or :obj:`func`, optional
        Constant value in the rMS norm which essentially controls when the norm allows a jump. This can be a
        constant number, a list of values (for multidimensional inputs, acting on the second dimension) or
        a function that is called passing a counter which keeps track of how many
        times the ``prox`` method has been invoked before and returns a scalar (or a list of)
        ``kappa`` to be used.

    Notes
    -----
    The :math:`rMS` proximal operator is defined as [1]_:

    .. math::
        \text{prox}_{\tau \text{rMS}}(x) =
        \begin{cases}
        \frac{1}{1+2\tau\alpha}x & \text{ if } & \vert x\vert \leq \sqrt{\frac{\kappa}{\alpha}(1 + 2\tau\alpha)} \\
        \kappa & \text{ else }
        \end{cases}.

    .. [1] Strekalovskiy, E., and D. Cremers, 2014, Real-time minimization of the piecewise smooth
            Mumford-Shah functional: European Conference on Computer Vision, 127–141.

    """
    def __init__(self, sigma=1., kappa=1.):
        super().__init__(None, False)
        self.sigma = sigma
        self.kappa = kappa
        self.count = 0

    def __call__(self, x):
        sigma = _current_sigma(self.sigma, self.count)
        kappa = _current_sigma(self.kappa, self.count)
        return np.minimum(sigma * np.linalg.norm(x) ** 2, kappa)

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        sigma = _current_sigma(self.sigma, self.count)
        kappa = _current_sigma(self.kappa, self.count)

        x = np.where(np.abs(x) <= np.sqrt(kappa / sigma * (1 + 2 * tau * sigma)), _l2(x, tau * sigma), x)
        return x
