__all__ = [
    "ModelUpdateCallback",
]

from typing import TYPE_CHECKING

from pylops.optimization.callback import Callbacks
from pylops.utils.typing import NDArray

if TYPE_CHECKING:
    from pyproximal.optimization.basesolver import Solver


class ModelUpdateCallback(Callbacks):  # type: ignore[misc]
    """Model update callback

    This callback can be used to stop the solver when each element of
    the model (i.e, solution) is updated below a certain threshold.

    It requires the solver to store the model of the previous iteration
    in a variable named ``xold``.

    Parameters
    ----------
    tol : :obj:`float`
        Absolute value of model update below which the solver
        will stop iterating. For example, if ``tol`` is 0.1, the solver
        will stop when the absolute value of each element of the difference
        between the current model is below below 0.1.

    """

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.stop = False

    def on_step_end(self, solver: "Solver", x: NDArray) -> None:
        if solver.ncp.abs(x - solver.xold).max() < self.tol:
            self.stop = True
