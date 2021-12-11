import numpy as np
from pyproximal.projection import SimplexProj


class L0BallProj():
    r"""L0 ball projection.

    Parameters
    ----------
    radius : :obj:`int`
        Radius

    Notes
    -----
    Given an L0 ball defined as:

    .. math::

        L0_{r} = \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}

    its orthogonal projection is computed by finding the r highest
    largest entries of :math:`\mathbf{x}` (in absolute value), keeping those
    and zero-ing all the other entries.
    Note the this is the proximal operator of the corresponding
    indicator function :math:`I_{L0_{r}}`.

    """
    def __init__(self, radius):
        self.radius = int(radius)

    def __call__(self, x):
        xshape = x.shape
        x = x.ravel()
        x[np.argsort(np.abs(x))[:-self.radius]] = 0
        return x.reshape(xshape)