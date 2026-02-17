import numpy as np
from pylops.utils.typing import NDArray


class EuclideanBallProj:
    r"""Euclidean ball projection.

    Parameters
    ----------
    center : :obj:`np.ndarray` or :obj:`float`
        Center of the ball
    radius : :obj:`float`
        Radius

    Notes
    -----
    Given an Euclidean ball defined as:

    .. math::

        \operatorname{Eucl}_{[c, r]} = \{ \mathbf{x}: l ||\mathbf{x} - \mathbf{c}||_2 \leq r \}

    its orthogonal projection is:

    .. math::

        P_{\operatorname{Eucl}_{[c, r]}} (\mathbf{x}) = \mathbf{c} + \frac{r}
        {max\{ ||\mathbf{x} - \mathbf{c}||_2^2, r\}}(\mathbf{x} - \mathbf{c})

    Note the this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{\operatorname{Eucl}_{[c, r]}}`.

    """

    def __init__(self, center: NDArray | float, radius: float):
        self.center = center
        self.radius = radius

    def __call__(self, x: NDArray) -> NDArray:
        x = self.center + self.radius / (
            max(float(np.linalg.norm(x - self.center)), self.radius)
        ) * (x - self.center)
        return x
