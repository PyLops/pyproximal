from typing import Any, Callable, List

from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray

from pyproximal.proximal._Dykstra import (
    _select_impl_by_arity,
    dykstra_two,
    parallel_dykstra_prox,
)
from pyproximal.ProxOperator import ProxOperator, _check_tau


class Sum(ProxOperator):
    r"""Proximal operator of the sum of proximable functions
    using Dykstra-like algorithm.

    Parameters
    ----------
    ops : :obj:`list`
        A list of proximable functions :math:`f_1, \ldots, f_m`.
    weights : :obj:`np.ndarray` or :obj:`list` or :obj:`None`, optional, default=None
        Weights :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`,
        used when :math:`m > 2`, or :math:`m = 2` and ``use_parallel=True``.
        Defaults to None, which means :math:`w_1 = \cdots = w_m = \frac{1}{m}.`
    niter : :obj:`int`, optional, default=1000
        The maximum number of iterations.
    tol : :obj:`float`, optional, default=1e-7
        Tolerance on change of the solution (used as stopping criterion).
        If ``tol=0``, run until ``niter`` is reached.
    use_parallel : :obj:`bool`, optional, default=False
        The parallel version is used when :math:`m > 2`,
        or :math:`m = 2` and `use_parallel=True`.
    use_original_tau : :obj:`bool`, optional, default=False
        Use the original value of :math:`\tau` (``True``)
        or the scaled version :math:`\tau_i = \tau / w_i` (``False``).

    Notes
    -----
    Given two functions :math:`f` and :math:`g`, or a set of proximable functions
    :math:`f_i` and corresponding weights :math:`w_i` for :math:`i=1, \ldots, m`,
    this class computes the proximal operator of the sum of two functions

    .. math:: \prox_{\tau (f + g)}

    using the Dykstra-like algorithm, or of the weighted sum of functions

    .. math:: \prox_{\tau \ \sum_{i=1}^m w_i f_i}

    using the parallel Dykstra-like algorithm.

    For :math:`m=2`, the proximal mapping :math:`\prox_{\tau (f + g)}(\mathbf{x})` of
    :math:`\mathbf{x}` is computed by the Dykstra-like algorithm [1]_, [2]_:

    * :math:`\mathbf{x}^0 = \mathbf{x}, \mathbf{p}^0 = \mathbf{q}^0 = \mathbf{0}`
    * for :math:`k = 1, \ldots`

      * :math:`\mathbf{y}^k = \prox_{\tau g}(\mathbf{x}^k + \mathbf{p}^k)`
      * :math:`\mathbf{p}^{k+1} = \mathbf{p}^k + \mathbf{x}^k - \mathbf{y}^k`
      * :math:`\mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{y}^k + \mathbf{q}^k)`
      * :math:`\mathbf{q}^{k+1} = \mathbf{q}^k + \mathbf{y}^k - \mathbf{x}^{k+1}`

    For :math:`m \ge 2`, the proximal mapping :math:`\prox_{\tau \sum_{i=1}^m w_i f_i}(\mathbf{x})`
    of :math:`\mathbf{x}` is computed by
    the parallel Dykstra-like algorithm [3]_, [4]_, [5]_,
    where :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`:

    * :math:`\mathbf{x}^0 = \mathbf{z}_1^0 = \cdots = \mathbf{z}_m^0 = \mathbf{x}`
    * for :math:`k = 1, \ldots`

      * :math:`\mathbf{x}^{k+1} = \sum_{i=1}^{m} w_i \prox_{\tau_i f_i} (\mathbf{z}_{i}^k)`
      * for :math:`i = 1, \ldots, m`

        * :math:`\mathbf{z}_{i}^{k+1} = \mathbf{z}_{i}^k + \mathbf{x}^{k+1} - \prox_{\tau_i f_i} (\mathbf{z}_{i}^k)`

    Note that :math:`\tau_i = \tau / w_i` if ``use_original_tau==False`` (default),
    otherwise :math:`\tau_i = \tau`.

    References
    ----------
    .. [1] Combettes, P.L., Pesquet, J.-C., 2011. Proximal Splitting Methods in
        Signal Processing, in Fixed-Point Algorithms for Inverse Problems in
        Science and Engineering, Springer, pp. 185-212. Algorithm 10.18.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [2] Bauschke, H.H., Combettes, P.L., 2008. A Dykstra-like algorithm for two
        monotone operators. Pacific Journal of Pitimization 4, 383-391.
        Theorem 3.3.
        http://www.ybook.co.jp/online-p/PJO/vol4/pjov4n3p383.pdf

    .. [3] Combettes, P.L., Pesquet, J.-C., 2011. Proximal Splitting Methods in
        Signal Processing, in Fixed-Point Algorithms for Inverse Problems in
        Science and Engineering, Springer, pp. 185-212. Algorithm 10.31.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [4] Combettes, P.L., Dũng, Đ., Vũ, B.C., 2011. Proximity for sums of composite
        functions. Journal of Mathematical Analysis and Applications 380, 680-688.
        Eq. (2.26)
        https://doi.org/10.1016/j.jmaa.2011.02.079
    .. [5] Combettes, P.L., 2009. Iterative Construction of the Resolvent of a Sum of
        Maximal Monotone Operators. Journal of Convex Analysis 16, 727-748.
        Theorem 4.2.
        https://www.heldermann.de/JCA/JCA16/JCA163/jca16044.htm

    See also
    --------
    projection.GenericIntersectionProj :
        The convex projection to the intersection of convex sets
        using Dykstra's algorithm.
    """

    def __init__(
        self,
        ops: List[ProxOperator],
        weights: NDArray | List[float] | None = None,
        niter: int = 1000,
        tol: float = 1e-7,
        use_parallel: bool = False,
        use_original_tau: bool = False,
    ) -> None:
        super().__init__(None, False)

        self.ops = ops
        self.niter = niter
        self.tol = tol
        self.use_original_tau = use_original_tau

        if weights is None:
            self.w = [1.0 / len(self.ops)] * len(self.ops)
        else:
            self.w = weights

        self._prox = _select_impl_by_arity(
            ops,
            use_parallel=use_parallel,
            single=self._single_prox,
            two=self._two_prox,
            more=self._more_prox,
        )

    def __call__(self, x: NDArray) -> bool | float:
        """Evaluate proximable functions

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector

        Returns
        -------
        :obj:`bool` or  :obj:`float`
            - return ``False`` immediately if any boolean-type ops is ``False``
            - return the sum of numeric-type ops values if all boolean-type ops are ``True``
            - return ``True`` if all ops are boolean-type (no numeric-type ops) and ``True``
        """
        # logic inspired by https://github.com/PyLops/pyproximal/issues/116
        ncp = get_array_module(x)

        def is_bool(v: bool | float) -> bool:
            return isinstance(v, (bool, ncp.bool_))

        prox_vals = [op(x) for op in self.ops]

        bools, vals = [], []
        for v in prox_vals:
            if is_bool(v):
                bools.append(v)
            else:
                vals.append(float(v))

        if bools and not all(bools):
            return False
        if vals:
            return sum(vals)
        return True

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        r"""compute :math:`\prox_{\tau \ f}(\mathbf{x})`` of :math:`\mathbf{x}`."""
        return self._prox(x, tau)

    def _single_prox(self, x0: NDArray, tau: float) -> NDArray:
        r"""Compute :math:`\prox_{\tau \ f}(\mathbf{x})` for :math:`m = 1`."""
        if len(self.ops) != 1:
            raise ValueError("len(ops) should be 1")

        return self.ops[0].prox(x0, tau)

    def _two_prox(self, x0: NDArray, tau: float) -> NDArray:
        r"""Compute :math:`\prox_{\tau \ f + g}(\mathbf{x})` for :math:`m = 2`."""
        if len(self.ops) != 2:
            raise ValueError("len(ops) should be 2")

        def bind_tau(
            prox: Callable[[NDArray, float], NDArray],
            tau: float,
        ) -> Callable[[NDArray], NDArray]:
            return lambda x: prox(x, tau)

        step1, step2 = [bind_tau(op.prox, tau) for op in self.ops]

        return dykstra_two(
            x0,
            step1,
            step2,
            niter=self.niter,
            tol=self.tol,
        )

    def _more_prox(self, x0: NDArray, tau: float) -> NDArray:
        r"""Compute :math:`\prox_{\tau \ \sum_{i=1}^m w_i f_i}(\mathbf{x})`
        for :math:`m \ge 2`.
        """

        def tau_policy(tau: float, w: NDArray | List[float]) -> List[float]:
            if self.use_original_tau:
                # legacy: all prox_i use the same tau
                return [tau] * len(w)
            # PPXA-like scaling: tau_i = T / w_i
            return [tau / wi for wi in w]

        if len(self.ops) < 2:
            raise ValueError("len(ops) should be 2 or larger")

        return parallel_dykstra_prox(
            x0,
            prox_ops=[op.prox for op in self.ops],
            weights=self.w,
            taus=tau_policy(tau, self.w),
            niter=self.niter,
            tol=self.tol,
        )
