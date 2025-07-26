import numpy as np


class HalfSpaceProj():
    r"""Half space projection.

    Parameters
    ----------
    w : :obj:`np.ndarray`
        Coefficients of the half space
    b : :obj:`float`
        bias of the half space

    Notes
    -----
    Given an half space defined as:

    .. math::

        \operatorname{H}_{[w, b]}
        = \{ \mathbf{x}: \mathbf{w}^T \mathbf{x} \leq b \}

    its orthogonal projection is:

    .. math::

        P_{\operatorname{H}_{[w, b]}} (\mathbf{x}) =
        \begin{cases}
        \mathbf{x},
            & \mathbf{w}^T \mathbf{x} \leq b \\
        \mathbf{x} - \frac{\mathbf{w}^T \mathbf{x} - b}{||\mathbf{w}||^2} \mathbf{w},
            & \text{otherwise}
        \end{cases}

    Note the this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{\operatorname{H}_{[w, b]}}`.

    """

    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.w_norm_sq = np.dot(w, w)

    def __call__(self, x):
        val = np.dot(self.w, x)
        if val <= self.b:
            return x
        else:
            factor = (val - self.b) / self.w_norm_sq
            x_proj = x - factor * self.w
            return x_proj
