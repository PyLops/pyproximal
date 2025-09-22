from typing import List, Callable
from pylops.utils.typing import NDArray

from pyproximal.proximal._dykstra_core import (
    dykstra_two,
    parallel_dykstra_projection,
    _select_impl_by_arity,
)


class GenericIntersectionProj():
    r"""The convex projection to the intersection of convex sets
    using Dykstra's algorithm.

    Parameters
    ----------
    projections : :obj:`list`
        A list of projection functions :math:`P_1, \ldots, P_m`.
    niter : :obj:`int`, optional, default=1000
        The maximum number of iterations.
    tol : :obj:`float`, optional, default=1e-6
        Tolerance on change of the solution (used as stopping criterion).
        If ``tol=0``, run until ``niter`` is reached.
    use_parallel : :obj:`bool`, optional, default=False
        If True, use the parallel version when $m=2$.

    Notes
    -----
    Given a collection of convex projections :math:`P_i` for :math:`i = 1, \ldots, m`,
    where each projection :math:`P_i(\mathbf{x})` maps a point :math:`\mathbf{x}`
    onto the convex set :math:`C_i`, the overall projection :math:`P_C(\mathbf{x})`
    of :math:`\mathbf{x}` onto the intersection of such sets
    :math:`C = \cap_{i=1}^m C_i` is computed using the Dykstra's algorithm.
    (:math:`C` should not be empty.)

    For :math:`m=2`, the projection :math:`P_C(\mathbf{x})` of :math:`\mathbf{x}` is computed
    by the Dykstra's algorithm [1]_, [2]_, [3]_:

    * :math:`\mathbf{x}_0 = \mathbf{x}, \mathbf{p}_0 = \mathbf{q}_0 = 0`,
    * for :math:`k = 1, 2, \ldots`

      * :math:`y_k = P_1(\mathbf{x}_k + \mathbf{p}_k)`
      * :math:`\mathbf{p}_{k+1} = \mathbf{x}_k + \mathbf{p}_k - y_k`
      * :math:`\mathbf{x}_{k+1} = P_2(y_k + \mathbf{q}_k)`
      * :math:`\mathbf{q}_{k+1} = y_k + \mathbf{q}_k - \mathbf{x}_{k+1}`

    For :math:`m \ge 2`, the projection :math:`P_C(\mathbf{x})` is computed
    by the parallel Dykstra's algorithm [5]_, [6]_. The following
    is taken from [4]_:

    * :math:`\mathbf{u}_m^{(0)} = \mathbf{x}, \mathbf{z}_1^{(0)} = \cdots = \mathbf{z}_m^{(0)} = 0`,
    * for :math:`k = 1, 2, \ldots`

      * :math:`\mathbf{u}_0^{(k)} = \mathbf{u}_m^{(k-1)}`
      * for :math:`i = 1, \ldots, m`

        * :math:`\mathbf{u}_i^{(k)} = P_i(\mathbf{u}_{i-1}^{(k)} + \mathbf{z}_i^{(k-1)})`
        * :math:`\mathbf{z}_i^{(k)} = \mathbf{z}_i^{(k-1)} + \mathbf{u}_{i-1}^{(k)} - \mathbf{u}_i^{(k)}`

    Note the this is the proximal operator of the corresponding
    indicator function
    (see :class:`pyproximal.GenericIntersectionProx` for details).

    References
    ----------
    .. [1] Bauschke, H.H., Borwein, J.M., 1994. Dykstra's Alternating
        Projection Algorithm for Two Sets. Journal of Approximation Theory 79,
        418-443. https://doi.org/10.1006/jath.1994.1136
        https://cmps-people.ok.ubc.ca/bauschke/Research/02.pdf
    .. [2] Bauschke, H.H., Burachik, R.S., Herman, D.B., Kaya, C.Y., 2020. On
        Dykstra's algorithm: finite convergence, stalling, and the method of
        alternating projections. Optim Lett 14, 1975-1987.
        https://doi.org/10.1007/s11590-020-01600-4
        https://arxiv.org/abs/2001.06747
    .. [3] Wikipedia, Dykstra's projection algorithm.
        https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm

    .. [4] Tibshirani, R.J., 2017. Dykstra's Algorithm, ADMM, and Coordinate
        Descent: Connections, Insights, and Extensions, NeurIPS2017. Eq.(4).
        https://proceedings.neurips.cc/paper_files/paper/2017/hash/5ef698cd9fe650923ea331c15af3b160-Abstract.html
    .. [5] Bauschke, H.H., Combettes, P.L., 2011. Convex Analysis and Monotone
        Operator Theory in Hilbert Spaces, Theorem 29.2, 1st ed, Springer.
        https://doi.org/10.1007/978-1-4419-9467-7
    .. [6] Bauschke, H.H., Lewis, A.S., 2000. Dykstras algorithm with bregman
        projections: A convergence proof. Optimization 48, 409-427.
        https://doi.org/10.1080/02331930008844513
        https://people.orie.cornell.edu/aslewis/publications/00-dykstras.pdf

    See also
    --------
    pyproximal.projection.IntersectionProj :
        The convex projection onto the intersection of a particular type of convex sets.
    pyproximal.GenericIntersectionProx :
        The corresponding indicator function.
    pyproximal.Sum :
        Proximal operator of a sum of two or more convex functions
        using Dykstra-like algorithm.
    """

    def __init__(
        self,
        projections: List[Callable[[NDArray], NDArray]],
        niter: int = 1000,
        tol: float = 1e-6,
        use_parallel: bool = False,
    ) -> None:

        self.projections = projections
        self.niter = niter
        self.tol = tol

        self._proj = _select_impl_by_arity(
            projections,
            use_parallel=use_parallel,
            single=self._single_proj,
            two=self._two_proj,
            more=self._more_proj,
        )

    def __call__(self, x: NDArray) -> NDArray:
        r"""compute projection :math:`P_C(\mathbf{x})` of :math:`\mathbf{x}`.
        """
        return self._proj(x)

    def _single_proj(self, x0: NDArray) -> NDArray:
        r"""Compute projection :math:`P_C(\mathbf{x})` for :math:`m=1`.
        """
        if len(self.projections) != 1:
            raise ValueError("len(projections) should be 1")

        return self.projections[0](x0)

    def _two_proj(
        self, x0: NDArray
    ) -> NDArray:
        r"""Compute projection :math:`P_C(\mathbf{x})` for :math:`m=2`.
        """
        if len(self.projections) != 2:
            raise ValueError("len(projections) should be 2")

        step1, step2 = self.projections

        return dykstra_two(
            x0, step1, step2,
            niter=self.niter,
            tol=self.tol,
        )

    def _more_proj(
            self, x0: NDArray
    ) -> NDArray:
        r"""Compute projection :math:`P_C(x)` for :math:`m \ge 2`.
        """
        if len(self.projections) < 2:
            raise ValueError("len(projections) should be 2 or larger")

        return parallel_dykstra_projection(
            x0,
            proj_ops=self.projections,
            niter=self.niter,
            tol=self.tol,
        )
