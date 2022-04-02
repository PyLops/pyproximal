import numpy as np
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator
from pyproximal.projection import BoxProj


class Box(ProxOperator):
    r"""Box proximal operator.

    Proximal operator of a Box: :math:`\operatorname{Box}_{[l, u]} = \{ x: l \leq x\leq u \}`.

    Parameters
    ----------
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound

    Notes
    -----
    As the Box is an indicator function, the proximal operator corresponds to
    its orthogonal projection (see :class:`pyproximal.projection.BoxProj` for
    details.

    """
    def __init__(self, lower=-np.inf, upper=np.inf):
        super().__init__(None, False)
        self.lower = lower
        self.upper = upper
        self.box = BoxProj(self.lower , self.upper)

    def __call__(self, x):
        return np.all((x > self.lower) & (x < self.upper)).astype(x.dtype)

    @_check_tau
    def prox(self, x, tau):
        return self.box(x)