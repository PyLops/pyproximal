from typing import Union

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.projection.Box import BoxProj
from pyproximal.ProxOperator import ProxOperator, _check_tau


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

    def __init__(
        self,
        lower: Union[float, NDArray] = -np.inf,
        upper: Union[float, NDArray] = np.inf,
    ) -> None:
        super().__init__(None, False)
        self.lower = lower
        self.upper = upper
        self.box = BoxProj(self.lower, self.upper)

    def __call__(self, x: NDArray) -> bool:
        return bool(np.all((x >= self.lower) & (x <= self.upper)))

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        return self.box(x)
