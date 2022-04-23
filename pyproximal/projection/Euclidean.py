import numpy as np
from pyproximal.projection import HyperPlaneBoxProj


class EuclideanBallProj():
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
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __call__(self, x):
        x = self.center +  \
            self.radius / (max(np.linalg.norm(x - self.center),
                               self.radius)) * (x - self.center)
        return x
