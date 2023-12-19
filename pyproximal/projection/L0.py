import numpy as np


class L0BallProj():
    r""":math:`L_0` ball projection.

    Parameters
    ----------
    radius : :obj:`int`
        Radius

    Notes
    -----
    Given an L0 ball defined as:

    .. math::

        L0_{r} = \{ \mathbf{x}: ||\mathbf{x}||_0 \leq r \}

    its orthogonal projection is computed by finding the :math:`r` highest
    largest entries of :math:`\mathbf{x}` (in absolute value), keeping those
    and zero-ing all the other entries.
    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{L0_{r}}`.

    """
    def __init__(self, radius):
        self.radius = int(radius)

    def __call__(self, x):
        xshape = x.shape
        xf = x.copy().flatten()
        xf[np.argsort(np.abs(xf))[:-self.radius]] = 0
        return xf.reshape(xshape)


class L01BallProj():
    r""":math:`L_{0,1}` ball projection.

    Parameters
    ----------
    radius : :obj:`int`
        Radius

    Notes
    -----
    Given an :math:`L_{0,1}` ball defined as:

    .. math::

        L_{0,1}^{r} =
        \{ \mathbf{x}: \text{count}([||\mathbf{x}_1||_1,
        ||\mathbf{x}_2||_1, ..., ||\mathbf{x}_1||_1] \ne 0) \leq r \}

    its orthogonal projection is computed by finding the :math:`r` highest
    largest entries of a vector obtained by applying the :math:`L_1` norm to each
    column of a matrix :math:`\mathbf{x}` (in absolute value), keeping those
    and zero-ing all the other entries.
    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{L_{0,1}^{r}}`.

    """
    def __init__(self, radius):
        self.radius = int(radius)

    def __call__(self, x):
        xc = x.copy()
        xf = np.linalg.norm(x, axis=0, ord=1)
        xc[:, np.argsort(np.abs(xf))[:-self.radius]] = 0
        return xc