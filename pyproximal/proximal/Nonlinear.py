from abc import ABC, abstractmethod
from typing import Tuple

from pylops.utils.typing import NDArray

from pyproximal.ProxOperator import ProxOperator, _check_tau


class Nonlinear(ABC, ProxOperator):
    r"""Nonlinear function proximal operator.

    Proximal operator for a generic nonlinear function :math:`f`. This is a
    template class which a user must subclass and implement the following
    methods:

    - ``fun``: a method evaluating the generic function :math:`f`.
    - ``grad``: a method evaluating the gradient of the generic function
      :math:`f`.
    - ``optimize``: a method that solves the optimization problem associated
      with the proximal operator of :math:`f`. Note that the
      ``_gradprox`` method must be used (instead of ``grad``) as this will
      automatically add the regularization term involved in the evaluation
      of the proximal operator.

    and optionally:

    - ``fungrad``: a method evaluating both the generic function :math:`f`
      and its gradient. If not implemented, the ``fun`` and ``grad`` methods
      will be called instead and their results returned.

    Parameters
    ----------
    x0 : :obj:`np.ndarray`
        Initial vector
    niter : :obj:`int`, optional
        Number of iterations of iterative scheme used to compute the proximal
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.

    Notes
    -----
    The proximal operator of a generic function requires solving the following
    optimization problem numerically

    .. math::

        prox_{\tau f} (\mathbf{x}) = arg \; min_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    which is done via the provided ``optimize`` method.

    """

    def __init__(
        self,
        x0: NDArray,
        niter: int = 10,
        warm: bool = True,
    ) -> None:
        super().__init__(None, True)
        self.niter = niter
        self.x0 = x0
        self.warm = warm

    @abstractmethod
    def fun(self, x: NDArray) -> float:
        pass

    @abstractmethod
    def grad(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def optimize(self) -> float:
        pass

    def __call__(self, x: NDArray) -> float:
        return self.fun(x)

    def _funprox(self, x: NDArray, tau: float) -> float:
        return self.fun(x) + 1.0 / (2 * tau) * float(((x - self.y) ** 2).sum())

    def _gradprox(self, x: NDArray, tau: float) -> NDArray:
        return self.grad(x) + 1.0 / tau * (x - self.y)

    def _fungradprox(self, x: NDArray, tau: float) -> Tuple[float, NDArray]:
        f, g = self.fungrad(x)
        f = f + 1.0 / (2 * tau) * ((x - self.y) ** 2).sum()
        g = g + 1.0 / tau * (x - self.y)
        return f, g

    def fungrad(self, x: NDArray) -> Tuple[float, NDArray]:
        f = self.fun(x)
        g = self.grad(x)
        return f, g

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        self.y = x
        self.tau = tau
        x = self.optimize()
        if self.warm:
            self.x0 = x
        return x
