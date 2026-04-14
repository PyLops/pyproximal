from __future__ import annotations

__all__ = ["Solver"]

import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any

from pylops.optimization.basesolver import Solver as pSolver
from pylops.optimization.callback import Callbacks

if TYPE_CHECKING:
    from pyproximal.ProxOperator import ProxOperator

_units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


class Solver(pSolver, metaclass=ABCMeta):
    r"""Solver

    This is a template class which a user must subclass when implementing a new solver.
    This class comprises of the following mandatory methods:

    - ``__init__``: initialization method the solver object is created and
      callbacks (if any present) are registered
    - ``memory_usage``: a method to compute upfront the memory used by each
      step of the solver
    - ``setup``: a method that is invoked to setup the solver, basically it will create
      anything required prior to applying a step of the solver
    - ``step``: a method applying a single step of the solver
    - ``run``: a method applying multiple steps of the solver
    - ``finalize``: a method that is invoked at the end of the optimization process. It can
      be used to do some final clean-up of the properties of the operator that we want
      to expose to the user
    - ``solve``: a method applying the entire optimization loop of the solver for a
      certain number of steps

    and optional methods:

    - ``_print_solver``: a method print on screen details of the solver (already implemented)
    - ``_print_setup``: a method print on screen details of the setup process
    - ``_print_step``: a method print on screen details of each step
    - ``_print_finalize``: a method print on screen details of the finalize process
    - ``callback``: a method implementing a callback function, which is called after
      every step of the solver

    Parameters
    ----------
    callbacks : :obj:`pylops.optimization.callback.Callbacks`
        Callbacks object used to implement custom callbacks

    Attributes
    ----------
    iiter : :obj:`int`
        Iteration counter.
    tstart : :obj:`float`
        Time at the start of the optimization process.
    tend : :obj:`float`
        Time at the end of the optimization process. Available
        only after ``finalize`` is called.
    telapsed : :obj:`float`
        Total time elapsed during the optimization process. Available
        only after ``finalize`` is called.

    """

    def __init__(
        self,
        callbacks: Callbacks = None,
    ) -> None:
        self.callbacks = callbacks
        self._registercallbacks()
        self.iiter = 0
        self.tstart = time.time()

    def _print_solver(self, text: str = "", nbar: int = 80) -> None:
        print(f"{type(self).__name__}" + text)
        print("-" * nbar)

    def _print_finalize(self, *args: Any, nbar: int = 80, **kwargs: Any) -> None:
        print(
            f"\nIterations = {self.iiter}        Total time (s) = {self.telapsed:.2f}"
        )
        print("-" * nbar + "\n")

    @abstractmethod
    def setup(
        self,
        proxf: ProxOperator | list[ProxOperator],
        proxg: ProxOperator | None = None,
        *args: Any,
        show: bool = False,
        **kwargs: Any,
    ) -> None:
        """Setup solver

        This method is used to setup the solver. Users can change the function signature
        by including any other input parameter required during the setup stage

        Parameters
        ----------
        proxf : :obj:`pyproximal."ProxOperator"` or :obj:`list`
            Proximal operator(s) to be used in the optimization
        proxg : :obj:`pyproximal."ProxOperator"`, optional
            Proximal operator for the regularization term (if None, no regularization is used)
        show : :obj:`bool`, optional
            Display setup log

        """
        pass

    @abstractmethod
    def solve(
        self,
        proxf: ProxOperator | list[ProxOperator],
        proxg: ProxOperator | None = None,
        *args,
        show: bool = False,
        **kwargs,
    ) -> Any:
        """Solve

        This method is used to run the entire optimization process. Users can change the
        function signature by including any other input parameter required by the solver

        Parameters
        ----------
        proxf : :obj:`pyproximal."ProxOperator"` or :obj:`list`
            Proximal operator(s) to be used in the optimization
        proxg : :obj:`pyproximal."ProxOperator"`, optional
            Proximal operator for the regularization term (if None, no regularization is used)
        show : :obj:`bool`, optional
            Display finalize log

        """
        pass
