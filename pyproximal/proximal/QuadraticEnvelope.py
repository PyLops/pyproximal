import numpy as np

from pylops.optimization.sparsity import _hardthreshold
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class QuadraticEnvelopeCard(ProxOperator):
    r"""Quadratic envelope of the :math:`\ell_0`-penalty.

    The :math:`\ell_0`-penalty is also known as the *cardinality function*, and the
    quadratic envelope :math:`\mathcal{Q}(\mu\|\cdot\|_0)` of it is defined as

    .. math::

        \mathcal{Q}(\mu\|\cdot\|_0)(x) = \sum_i \left(\mu - \frac{1}{2}\max(0, \sqrt{2\mu} - |x_i|)^2\right)

    where :math:`\mu \geq 0`.

    Parameters
    ----------
    mu : :obj:`float`
        Threshold parameter.

    See Also
    --------
    QuadraticEnvelopeCardIndicator: Quadratic envelope of the indicator function of :math:`\ell_0`-penalty

    Notes
    -----
    The terminology *quadratic envelope* was coined in [1]_, however, the rationale has
    been used earlier, e.g. in [2]_. In a general setting, the quadratic envelope
    :math:`\mathcal{Q}(f)(x)` is defined such that

    .. math::

        \left(f(x) + \frac{1}{2}\|x-y\|_2^2\right)^{**} = \mathcal{Q}(f)(x) + \frac{1}{2}\|x-y\|_2^2

    where :math:`g^{**}` denotes the bi-conjugate of :math:`g`, which is the l.s.c.
    convex envelope of :math:`g`.

    There is no closed-form expression for :math:`\mathcal{Q}(f)(x)` given an arbitrary
    function :math:`f`. However, for certain special cases, such as in the case of the
    cardinality function, such expressions do exist.

    The proximal operator is given by

    .. math::

        \prox_{\tau\mathcal{Q}(\mu\|\cdot\|_0)}(x) =
        \begin{cases}
        x_i, & |x_i| \geq \sqrt{2 \mu} \\
        \frac{x_i-\tau\sqrt{2\mu}\sgn(x_i)}{1-\tau}, & \tau\sqrt{2\mu} < |x_i| < \sqrt{2 \mu} \\
        0, & |x_i| \leq \tau\sqrt{2 \mu}
        \end{cases}

    By inspecting the structure of the proximal operator it is clear that large values
    are unaffected, whereas smaller ones are penalized partially or completely. Such
    properties are desirable to counter the effect of *shrinking bias* observed with
    e.g. the :math:`\ell_1`-penalty. Note that in the limit :math:`\tau=1` this becomes
    the hard thresholding with threshold :math:`\sqrt{2\mu}`. It should also be noted
    that this proximal operator is identical to the Minimax Concave Penalty (MCP)
    proposed in [3]_.

    References
    ----------
    .. [1] Carlsson, M. "On Convex Envelopes and Regularization of Non-convex
        Functionals Without Moving Global Minima", In Journal of Optimization Theory
        and Applications, 183:66–84, 2019.
    .. [2] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.
    .. [3] Zhang et al. "Nearly unbiased variable selection under minimax concave
        penalty", In the Annals of Statistics, 38(2):894–942, 2010.

    """

    def __init__(self, mu):
        super().__init__(None, False)
        self.mu = mu

    def __call__(self, x):
        return np.sum(self.elementwise(x))

    def elementwise(self, x):
        return self.mu - 0.5 * np.maximum(0, np.sqrt(2 * self.mu) - np.abs(x)) ** 2

    @_check_tau
    def prox(self, x, tau):
        r = np.abs(x)
        idx = r < np.sqrt(2 * self.mu)
        if tau >= 1:
            r[idx] = 0
        else:
            r[idx] = np.maximum(0, (r[idx] - tau * np.sqrt(2 * self.mu)) / (1 - tau))
        return r * np.sign(x)


class QuadraticEnvelopeCardIndicator(ProxOperator):
    r"""Quadratic envelope of the indicator function of the :math:`\ell_0`-penalty.

    The :math:`\ell_0`-penalty is also known as the *cardinality function*, and the
    indicator function :math:`\mathcal{I}_{r_0}` is defined as

    .. math::

        \mathcal{I}_{r_0}(\mathbf{x}) =
        \begin{cases}
        0, & \mathbf{x}\leq r_0 \\
        \infty, & \text{otherwise}
        \end{cases}

    Let :math:`\tilde{\mathbf{x}}` denote the vector :math:`\mathbf{x}` resorted such that the
    sequence :math:`(\tilde{x}_i)` is non-increasing. The quadratic envelope
    :math:`\mathcal{Q}(\mathcal{I}_{r_0})` can then be written as

    .. math::

        \mathcal{Q}(\mathcal{I}_{r_0})(x) =
        \frac{1}{2k^*}\left(\sum_{i>r_0-k^*}|\tilde{x}_i|\right)^2
        - \frac{1}{2}\left(\sum_{i>r_0-k^*}|\tilde{x}_i|\right)^2

    where :math:`r_0 \geq 0` and :math:`k^* \leq r_0`, see [3]_ for details. There are
    other, equivalent ways, of expressing this penalty, see e.g. [1]_ and [2]_.

    Parameters
    ----------
    r0 : :obj:`int`
        Threshold parameter.

    See Also
    --------
    QuadraticEnvelopeCard: Quadratic envelope of the :math:`\ell_0`-penalty

    Notes
    -----
    The terminology *quadratic envelope* was coined in [1]_, however, the rationale has
    been used earlier, e.g. in [2]_. In a general setting, the quadratic envelope
    :math:`\mathcal{Q}(f)(x)` is defined such that

    .. math::

        \left(f(x) + \frac{1}{2}\|x-y\|_2^2\right)^{**} = \mathcal{Q}(f)(x) + \frac{1}{2}\|x-y\|_2^2

    where :math:`g^{**}` denotes the bi-conjugate of :math:`g`, which is the l.s.c.
    convex envelope of :math:`g`.

    There is no closed-form expression for :math:`\mathcal{Q}(f)(x)` given an arbitrary
    function :math:`f`. However, for certain special cases, such as in the case of the
    indicator function of the cardinality function, such expressions do exist.

    The proximal operator does not have a closed-form, and we refer to [1]_ for more details.
    Note that this is a non-separable penalty.

    References
    ----------
    .. [1] Carlsson, M. "On Convex Envelopes and Regularization of Non-convex
        Functionals Without Moving Global Minima", In Journal of Optimization Theory
        and Applications, 183:66–84, 2019.
    .. [2] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.
    .. [3] Andersson et al. "Convex envelopes for fixed rank approximation", In
        Optimization Letters, 11:1783–1795, 2017.

    """

    def __init__(self, r0):
        super().__init__(None, False)
        self.r0 = r0

    def __call__(self, x):
        if x.size <= self.r0 or np.count_nonzero(x) <= self.r0:
            return 0
        xs = np.sort(np.abs(x))[::-1]
        sums = np.cumsum(xs[::-1])
        sums = sums[-self.r0:] / np.arange(1, self.r0 + 1)
        tmp = np.diff(sums) > 0
        k_star = np.argmax(tmp)
        if k_star == 0 and not tmp[k_star]:
            k_star = self.r0 - 1
        return 0.5 * ((k_star + 1) * sums[k_star] ** 2 - np.sum(xs[self.r0-k_star-1:] ** 2))

    @_check_tau
    def prox(self, y, tau):
        rho = 1 / tau
        if rho <= 1:
            return _hardthreshold(y, tau)
        if y.size <= self.r0:
            return y

        r = np.abs(y)
        theta = np.sign(y)
        id = np.argsort(-r, kind='quicksort')
        rsort = r[id]
        idinv = np.zeros_like(id)
        idinv[id] = np.arange(r.size)
        rnew = np.concatenate((rsort[:self.r0], rho * rsort[self.r0:]))

        if rho * rsort[self.r0] < rsort[self.r0 - 1]:
            x = rnew
            x = x[idinv]
            x = x * theta
        else:
            j = np.min(np.where(rnew <= rnew[self.r0])[0])
            l = np.max(np.where(rnew >= rnew[self.r0 - 1])[0])
            z = np.sort(rnew[j:l + 1])[::-1]
            z1 = z[0]
            for z2 in z[1:]:
                s = (z1 + z2) / 2
                temp = np.where(rnew <= s)[0]
                j1 = np.min(temp)
                temp = np.where(rnew >= s)[0]
                l1 = np.max(temp)
                sI = (rho * sum(rsort[j1:l1 + 1])) / ((self.r0 - j1) * rho + (l1 + 1 - self.r0) * 1)
                if z2 <= sI <= z1:
                    x = np.concatenate((np.maximum(rnew[:self.r0], sI), np.minimum(rnew[self.r0:], sI)))
                    x = x[idinv]
                    x = x * theta
                    break
                z1 = z2

        return (rho * y - x) / (rho - 1)


class QuadraticEnvelopeRankL2(ProxOperator):
    r"""Quadratic envelope of the rank function with an L2 misfit term.

    The penalty :math:`p` is given by

     .. math::

        p(X) = \mathcal{R}_{r_0}(X) + \frac{1}{2}\|X - M\|_F^2

    where :math:`\mathcal{R}_{r_0}` is the quadratic envelope of the hard-rank function.

    Parameters
    ----------
    dim : :obj:`tuple`
        Size of input matrix :math:`X`.
    r0 : :obj:`int`
        Threshold parameter, encouraging matrices with rank lower than or equal to r0.
    M : :obj:`numpy.ndarray`
        L2 misfit term (must be the same size as the input matrix).

    See Also
    --------
    SingularValuePenalty: Proximal operator of a penalty acting on the singular values
    QuadraticEnvelopeCardIndicator: Quadratic envelope of the indicator function of :math:`\ell_0`-penalty

    Notes
    -----
    The proximal operator solves the minimization problem

        .. math::
            \argmin_Z \mathcal{R}_{r_0}(Z) + \frac{1}{2}\|Z - M\|_F^2 + \frac{1}{2\tau}\| Z - X \|_F^2

    which is a convex-concave min-max problem, see [1]_ for details.

    References
    ----------
    .. [1] Larsson, V. and Olsson, C. "Convex Low Rank Approximation", In International
        Journal of Computer Vision (IJCV), 120:194–214, 2016.

    """

    def __init__(self, dim, r0, M):
        super().__init__(None, False)
        self.dim = dim
        self.r0 = r0
        self.M = M.copy()
        self.penalty = QuadraticEnvelopeCardIndicator(r0)

    def __call__(self, x):
        X = x.reshape(self.dim)
        eigs = np.linalg.eigvalsh(X.T @ X)
        eigs[eigs < 0] = 0  # ensure all eigenvalues at positive
        return np.sum(self.penalty(np.sqrt(eigs))) + 0.5 * np.linalg.norm(X - self.M, 'fro')

    @_check_tau
    def prox(self, x, tau):
        rho = 1 / tau
        P = x.reshape(self.dim)

        Y = (self.M + rho * P) / (1 + rho)
        U, yk, Vh = np.linalg.svd(Y, full_matrices=False)
        n = yk.size
        r = np.concatenate((yk[:self.r0], (1 + rho) * yk[self.r0:]))
        ind = np.argsort(r, kind='quicksort')
        p = r[ind]

        a = (n - self.r0) / rho
        b = (rho + 1) / rho * np.sum(yk[self.r0:])

        # Base case
        zk = yk.copy()
        zk[self.r0:] = (1 + rho) * yk[self.r0:]

        for k, ii in enumerate(ind):

            if ii < self.r0:
                a = a + (rho + 1) / rho
                b = b + (rho + 1) / rho * yk[ii]
            else:
                a = a - 1 / rho
                b = b - (rho + 1) / rho * yk[ii]

            if a == 0:
                continue

            s = b / a

            if p[k] <= s <= p[k + 1]:
                zk = np.maximum(s, yk)
                zk[self.r0:] = np.minimum(s, (1 + rho) * yk[self.r0:])
                break

        Z = np.dot(U * zk, Vh)
        X = P + (self.M - Z) / rho
        return X.ravel()
