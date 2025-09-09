from typing import List, Callable, Any

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau
from pyproximal.projection import DykstrasProjection


class DykstrasProjectionProx(ProxOperator):
    r"""The proximal operator corresponding to the convex projection to the
    intersection of convex sets using Dykstra's algorithm.

    Parameters
    ----------
    projections : :obj:`List[Callable[[np.ndarray], np.ndarray]]`
        A list of projection functions :math:`P_1, \ldots, P_m`.
    max_iter : :obj:`int`, optional, default=100
        The maximum number of iterations.
    tol : :obj:`float`, optional, default=1e-6
        Torrelance to stop the iteration.
    use_parallel : :obj:`bool`, optional, default=False
        If True, use the parallel version when $m=2$.

    Notes
    -----
    As the intersection of convex sets is an indicator function,
    the proximal operator corresponds to its convex projection
    (see :class:`pyproximal.projection.DykstrasProjection` for details).


    Examples
    --------
    >>> import numpy as np
    >>> from pyproximal.projection import (
    ...         BoxProj,
    ...         EuclideanBallProj,
    ... )
    >>> from pyproximal.proximal import DykstrasProjectionProx

    >>> circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
    >>> circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
    >>> circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
    >>> box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

    >>> projections = [circle_1, circle_2, circle_3, box]
    >>> dykstra_prox = DykstrasProjectionProx(projections)

    >>> rng = np.random.default_rng(10)
    >>> x = rng.normal(0., 3.5, size=2)

    >>> print("x            =", x)
    x            = [-3.86168457 -2.53758624]
    >>> dykstra_prox(x)  # x is outside
    False

    >>> xp = dykstra_prox.prox(x, 1.0)  # DykstrasProjection
    >>> print("x projection =", xp)
    x projection = [-2.42308423 -0.87363268]
    >>> dykstra_prox(xp)  # x is inside
    True


    See also
    --------
    pyproximal.projection.DykstrasProjection :
        The corresponding convex projection.

    """

    def __init__(
        self,
        projections: List[Callable[[NDArray], NDArray]],
        max_iter: int = 100,
        tol: float = 1e-6,
        use_parallel: bool = False,
    ) -> None:
        super().__init__(None, False)
        self.projections = projections

        # The tolerance for the indicator function is set to 10 times larger
        # than the tolerance used in Dykstra's projection. This is because
        # using the same tolerance does not guarantee that the condition
        # will hold even after the convergence of Dykstra's algorithm.
        self.tol = tol * 10

        self.dykstras_projection = \
            DykstrasProjection(
                projections=self.projections,
                max_iter=max_iter,
                tol=tol,
                use_parallel=use_parallel,
            )

    def __call__(self, x: NDArray) -> bool:
        return all(np.abs(x - proj(x)).max() < self.tol
                   for proj in self.projections)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.dykstras_projection(x)
