import numpy as np
from scipy.optimize import bisect


class BoxProj():
    r"""Box orthogonal projection.

    Parameters
    ----------
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound

    Notes
    -----
    Given a Box set defined as:

    .. math::

        \operatorname{Box}_{[l, u]} = \{ x: l \leq x\leq u \}

    its orthogonal projection is:

    .. math::

        P_{\operatorname{Box}_{[l, u]}} (x_i) = min\{ max \{x_i, l_i\}, u_i \} =
        \begin{cases}
        l_i, & x_i < l_i\\
        x_i,& l_i \leq x_i \leq u_i \\
        u_i,  & x_i > u_i\\
        \end{cases} \quad \forall i

    Note that this is the proximal operator of the corresponding
    indicator function :math:`\mathcal{I}_{\operatorname{Box}_{[l, u]}}`.

    """
    def __init__(self, lower=-np.inf, upper=np.inf):
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        x = np.minimum(np.maximum(x, self.lower), self.upper)
        return x


class HyperPlaneBoxProj():
    r"""Orthogonal projection of the intersection between a Hyperplane and a
    Box.

    Parameters
    ----------
    coeffs : :obj:`np.ndarray`
        Vector of coefficients used in the definition of the hyperplane
    scalar : :obj:`float`
        Scalar used in the definition of the hyperplane
    lower : :obj:`float` or :obj:`np.ndarray`, optional
        Lower bound of Box
    upper : :obj:`float` or :obj:`np.ndarray`, optional
        Upper bound of Box
    maxiter : :obj:`int`, optional
        Maximum number of iterations used by :func:`scipy.optimize.bisect`
    xtol : :obj:`float`, optional
        Absolute tolerance of :func:`scipy.optimize.bisect`

    Notes
    -----
    Given the definition of an Hyperplane:

    .. math::

        H_{c,b} = \{ \mathbf{x}: \mathbf{c}^T \mathbf{x} = b\}

    that of a Box (see :class:`pyproximal.projection.Box.BoxProj`), the
    intersection between the two can be written as:

    .. math::

        C = Box_{[l, u]} \cap H_{c,b} =
        \{ \mathbf{x}: \mathbf{c}^T \mathbf{x} = b , \; l \leq x_i \leq u \}

    The orthogonal projection of such intersection is given by:

    .. math::

        P_C = P_{Box_{[l, u]}} (\mathbf{x} - \mu^* \mathbf{c})

    where :math:`\mu` is obtained by solving the following equation by
    bisection

    .. math::

        f(\mu) = \mathbf{c}^T P_{Box_{[l, u]}} (\mathbf{x} -
        \mu \mathbf{c}) - b

    """
    def __init__(self, coeffs, scalar, lower=-np.inf, upper=np.inf,
                 maxiter=100, xtol=1e-5):
        self.coeffs = coeffs.ravel()
        self.scalar = scalar
        self.lower = lower
        self.upper = upper
        self.maxiter = maxiter
        self.xtol = xtol
        self.box = BoxProj(lower, upper)

    def __call__(self, x):
        """Apply HyperPlaneBoxProj projection

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector

        """
        def fun(mu, x):
            return np.dot(self.coeffs, self.box(x - mu * self.coeffs)) - \
                   self.scalar

        xshape = x.shape
        x = x.ravel()

        # identify brackets for bisect ensuring that the evaluated fun
        # has different sign
        bisect_lower = -1
        while fun(bisect_lower, x) < 0:
            bisect_lower *= 2

        bisect_upper = 1
        while fun(bisect_upper, x) > 0:
            bisect_upper *= 2

        # find optimal mu
        mu = bisect(lambda mu: fun(mu, x), bisect_lower, bisect_upper,
                    maxiter=self.maxiter, xtol=self.xtol)

        # compute projection
        y = self.box(x - mu * self.coeffs)
        return y.reshape(xshape)