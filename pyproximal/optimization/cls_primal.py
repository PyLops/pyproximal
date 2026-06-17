__all__ = [
    "ProximalPoint",
    "ProximalGradient",
    "AndersonProximalGradient",
    "GeneralizedProximalGradient",
    "HQS",
    "ADMM",
    "ADMML2",
    "LinearizedADMM",
    "TwIST",
    "DouglasRachfordSplitting",
]

from collections.abc import Sequence
from math import sqrt
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import pylops
from pylops.optimization.callback import Callbacks
from pylops.optimization.leastsquares import regularized_inversion
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray

from pyproximal.optimization.basesolver import Solver
from pyproximal.proximal import L2
from pyproximal.ProxOperator import ProxOperator
from pyproximal.utils.bilinear import BilinearOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


# need to check pylops version since _callback_stop
# is only available in pylops>=2.6.0
sp_version = pylops.__version__.split(".")
if int(sp_version[0]) < 2 or (int(sp_version[0]) == 2 and int(sp_version[1]) < 6):

    def _callback_stop(callbacks: Sequence[Callbacks]) -> bool:
        return False
else:
    from pylops.optimization.callback import _callback_stop  # type: ignore[no-redef]


def _backtracking(
    x: NDArray,
    tau: float,
    proxf: ProxOperator,
    proxg: ProxOperator,
    epsg: float,
    beta: float = 0.5,
    niterback: int = 10,
) -> tuple[NDArray, float]:
    r"""Backtracking

    Line-search algorithm for finding step sizes in proximal algorithms when
    the Lipschitz constant of the operator is unknown (or expensive to
    estimate).

    """

    def ftilde(x: NDArray, y: NDArray, f: ProxOperator, tau: float) -> float:
        xy = x - y
        return float(
            f(y) + np.dot(f.grad(y), xy) + (1.0 / (2.0 * tau)) * np.linalg.norm(xy) ** 2
        )

    iiterback = 0
    while iiterback < niterback:
        z = proxg.prox(x - tau * proxf.grad(x), epsg * tau)
        ft = ftilde(z, x, proxf, tau)
        if proxf(z) <= ft:
            break
        tau *= beta
        iiterback += 1
    return z, tau


def _x0z0_init(
    x0: NDArray | None,
    z0: NDArray | None,
    Op: Optional["LinearOperator"] = None,
    z0name: str | None = "z0",
    Opname: str | None = "Op",
) -> tuple[NDArray, NDArray]:
    r"""Initialize x0 and z0

    Initialize x0 and z0 using the following convention.

    For ``Op=None``:
    - if both are provided, they are simply returned;
    - if only one is provided (the other is ``None``), the one provided
      is copied to the other one.

    For ``Op!=None``, ``x0`` must be provided, and:
    - if both are provided, they are simply returned;
    - if ``z0`` is not provided, set to ``Op @ x0``.

    Parameters
    ----------
    x0 : :obj:`numpy.ndarray`
        Initial vector
    z0 : :obj:`numpy.ndarray`
        Initial auxiliary vector
    Op : :obj:`pylops.LinearOperator`, optional
        Linear Operator to apply to ``x0``
    z0name : :obj:`str`, optional
        Name to display in error message instead of ``z0``
    Opname : :obj:`str`, optional
        Name to display in error message instead of ``Op``

    """
    if x0 is None and z0 is None:
        msg = f"Both x0 or {z0name} are None, provide either of them or both"
        raise ValueError(msg)

    if Op is None:
        if x0 is None:
            x0 = z0.copy()  # type: ignore[union-attr]
        elif z0 is None:
            z0 = x0.copy()
    else:
        if x0 is None:
            msg = f"x0 must be provided when {Opname} is also provided"
            raise ValueError(msg)
        elif z0 is None:
            z0 = Op @ x0
    return x0, z0


class ProximalPoint(Solver):
    r"""Proximal point algorithm

    Solves the following minimization problem using Proximal point algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x})

    where :math:`f(\mathbf{x})` is any convex function that has a known
    proximal operator.

    Notes
    -----
    The Proximal point algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^k)

    """

    pf: float

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=60)

        strpar = f"Proximal operator: {type(self.prox).__name__}"
        if self.niter is not None:
            strpar1 = (
                f"tau = {self.tau:6e}\ttol = {str(self.tol)}\tniter = {self.niter}"
            )
        else:
            strpar1 = f"tau = {self.tau:6e}\ttol = {str(self.tol)}"
        print(strpar)
        print(strpar1)
        print("-" * 60 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]                f"
        else:
            head1 = "    Itn              x[0]                     f"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        if self.tol is None:
            self.pf = self.prox(x)
        strx = f"{x[0]:1.2e}        " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = f"{self.iiter:6g}        " + strx + f"{self.pf:11.4e}"
        print(msg)

    def setup(  # type: ignore[override]
        self,
        prox: "ProxOperator",
        x0: NDArray,
        tau: float,
        niter: int | None = None,
        tol: float = 1e-4,
        show: bool = False,
    ) -> NDArray:
        r"""Setup solver

        Parameters
        ----------
        prox : :obj:`pyproximal."ProxOperator"`
            Proximal operator
        x0 : :obj:`numpy.ndarray`
            Initial guess
        tau : :obj:`float`
            Positive scalar weight
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme (default to ``None``
            in case a user wants to manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess

        """
        self.prox = prox
        self.x0 = x0
        self.tau = tau
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # create variables to track the objective function and iterations
        self.pf, self.pfold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x0

    def step(self, x: NDArray, show: bool = False) -> NDArray:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            proximal point algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector

        """
        x = self.prox.prox(x, self.tau)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfold = self.pf
            self.pf = self.prox(x)
            if np.abs(1.0 - self.pf / self.pfold) < self.tol:
                self.tolbreak = True

        self.iiter += 1
        if show:
            self._print_step(x)
        if self.tol is not None or show:
            self.cost.append(float(self.pf))
        return x

    def run(
        self,
        x: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the proximal point algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x = self.step(x, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x

    def solve(  # type: ignore[override]
        self,
        prox: "ProxOperator",
        x0: NDArray,
        tau: float,
        niter: int = 10,
        tol: float = 1e-4,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        prox : :obj:`pyproximal."ProxOperator"`
            Proximal operator
        x0 : :obj:`numpy.ndarray`, optional
            Initial guess
        tau : :obj:`float`
            Positive scalar weight
        niter : :obj:`int`, optional
            Number of iterations
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x = self.setup(prox=prox, x0=x0, tau=tau, niter=niter, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(60, show)
        return x, self.iiter, self.cost


class ProximalGradient(Solver):
    r"""Proximal gradient (optionally accelerated)

    Solves the following minimization problem using (Accelerated) Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Notes
    -----
    The Proximal gradient algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \mathbf{y}^k + \eta (\prox_{\tau^k \epsilon g}(\mathbf{y}^k -
        \tau^k \nabla f(\mathbf{y}^k)) - \mathbf{y}^k) \\
        \mathbf{y}^{k+1} = \mathbf{x}^k + \omega^k
        (\mathbf{x}^k - \mathbf{x}^{k-1})

    where at each iteration :math:`\tau^k` can be estimated by back-tracking
    as follows:

    .. math::

        \begin{aligned}
        &\tau = \tau^{k-1} &\\
        &repeat \; \mathbf{z} = \prox_{\tau \epsilon g}(\mathbf{x}^k -
        \tau \nabla f(\mathbf{x}^k)), \tau = \beta \tau \quad if \;
        f(\mathbf{z}) \leq \tilde{f}_\tau(\mathbf{z}, \mathbf{x}^k) \\
        &\tau^k = \tau, \quad \mathbf{x}^{k+1} = \mathbf{z} &\\
        \end{aligned}

    where :math:`\tilde{f}_\tau(\mathbf{x}, \mathbf{y}) = f(\mathbf{y}) +
    \nabla f(\mathbf{y})^T (\mathbf{x} - \mathbf{y}) +
    1/(2\tau)||\mathbf{x} - \mathbf{y}||_2^2`.

    Different accelerations are provided:

    - ``acceleration=None``: :math:`\omega^k = 0`;
    - ``acceleration=vandenberghe`` [1]_: :math:`\omega^k = k / (k + 3)` for `
    - ``acceleration=fista``: :math:`\omega^k = (t_{k-1}-1)/t_k` where
      :math:`t_k = (1 + \sqrt{1+4t_{k-1}^{2}}) / 2` [2]_

    .. [1] Vandenberghe, L., "Fast proximal gradient methods", 2010.
    .. [2] Beck, A., and Teboulle, M. "A Fast Iterative Shrinkage-Thresholding
       Algorithm for Linear Inverse Problems", SIAM Journal on
       Imaging Sciences, vol. 2, pp. 183-202. 2009.

    """

    def _print_setup(self, epsg_print: str, xcomplex: bool = False) -> None:
        self._print_solver(nbar=81)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
        )

        strpar1 = f"tau = {self.tau:4.2e}\t\tbacktrack = {self.backtracking}"
        if self.niter is not None:
            strpar2 = f"beta = {self.beta}\tepsg = {epsg_print}\t\tniter = {self.niter}\t\ttol = {str(self.tol)}"
        else:
            strpar2 = (
                f"beta = {self.beta}\t epsg = {epsg_print}\t\ttol = {str(self.tol)}"
            )
        strpar3 = f"niterback = {self.niterback}\t\tacceleration = {self.acceleration}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print(strpar3)
        print("-" * 81 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g       J=f+eps*g       tau"
        else:
            head1 = "    Itn              x[0]                  f           g       J=f+eps*g       tau"
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + np.sum(self.epsg[self.iiter - 1] * pg)
        x0 = to_numpy(x[0]) if x.ndim == 1 else to_numpy(x[0, 0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}  {self.tau:11.2e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        epsg: float | NDArray = 1.0,
        tau: float | None = None,
        backtracking: bool = False,
        beta: float = 0.5,
        eta: float = 1.0,
        niter: int = 10,
        niterback: int = 100,
        acceleration: str | None = None,
        tol: float | None = None,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function (must have ``grad`` implemented)
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor of g function
        tau : :obj:`float` or :obj:`numpy.ndarray`, optional
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
            backtracking is used to adaptively estimate the best tau at each
            iteration. Finally, note that :math:`\tau` can be chosen to be a vector
            when dealing with problems with multiple right-hand-sides
        backtracking : :obj:`bool`, optional
            Force backtracking, even if ``tau`` is not equal to ``None``. In this case
            the chosen ``tau`` will be used as the initial guess in the first
            step of backtracking
        beta : :obj:`float`, optional
            Backtracking parameter (must be between 0 and 1)
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 1, 0 excluded).
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        niterback : :obj:`int`, optional
            Max number of iterations of backtracking
        acceleration : :obj:`str`, optional
            Acceleration (``None``, ``vandenberghe`` or ``fista``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        y : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        """
        self.proxf = proxf
        self.proxg = proxg
        self.x0 = x0
        self.backtracking = backtracking
        self.beta = beta
        self.eta = eta
        self.niter = niter
        self.niterback = niterback
        self.tol = tol

        self.ncp = get_array_module(x0)

        # check if epgs is a vector
        self.epsg = np.asarray(epsg, dtype=float)
        if self.epsg.size == 1:
            self.epsg = epsg * np.ones(niter)
            epsg_print = str(self.epsg[0])
        else:
            epsg_print = "Multi"

        # set tau
        self.tau = tau
        if tau is None:
            self.backtracking = True
            self.tau = 1.0

        # check acceleration
        if acceleration in [None, "None", "vandenberghe", "fista"]:
            self.acceleration = acceleration
        else:
            msg = "Acceleration should be None, vandenberghe or fista"
            raise NotImplementedError(msg)

        # initialize solver
        x = x0.copy()
        y = x.copy()

        # for accelaration
        self.t = 1.0

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(epsg_print, np.iscomplexobj(x0))
        return x, y

    def step(
        self, x: NDArray, y: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            proximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            proximal gradient algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        xold = x.copy()

        # proximal step
        if not self.backtracking:
            if self.eta == 1.0:
                x = self.proxg.prox(
                    y - self.tau * self.proxf.grad(y), self.epsg[self.iiter] * self.tau
                )
            else:
                x = x + self.eta * (
                    self.proxg.prox(
                        x - self.tau * self.proxf.grad(x),
                        self.epsg[self.iiter] * self.tau,
                    )
                    - x
                )
        else:
            x, self.tau = _backtracking(
                y,
                cast(float, self.tau),
                self.proxf,
                self.proxg,
                self.epsg[self.iiter],
                beta=self.beta,
                niterback=self.niterback,
            )
            if self.eta != 1.0:
                x = x + self.eta * (
                    self.proxg.prox(
                        x - self.tau * self.proxf.grad(x),
                        self.epsg[self.iiter] * self.tau,
                    )
                    - x
                )

        # update internal parameters for bilinear operator
        if isinstance(self.proxf, BilinearOperator):
            self.proxf.updatexy(x)

        # update y
        if self.acceleration == "vandenberghe":
            omega = self.iiter / (self.iiter + 3)
        elif self.acceleration == "fista":
            told = self.t
            self.t = (1.0 + np.sqrt(1.0 + 4.0 * self.t**2)) / 2.0
            omega = (told - 1.0) / self.t
        else:
            omega = 0
        y = x + omega * (x - xold)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + np.sum(self.epsg[self.iiter] * pg)
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, y

    def run(
        self,
        x: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the proximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the proximal gradient algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, y = self.step(x, y, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        epsg: float | NDArray = 1.0,
        tau: float | None = None,
        backtracking: bool = False,
        beta: float = 0.5,
        eta: float = 1.0,
        niter: int = 10,
        niterback: int = 100,
        acceleration: str | None = None,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function (must have ``grad`` implemented)
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor of g function
        tau : :obj:`float` or :obj:`numpy.ndarray`, optional
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. When ``tau=None``,
            backtracking is used to adaptively estimate the best tau at each
            iteration. Finally, note that :math:`\tau` can be chosen to be a vector
            when dealing with problems with multiple right-hand-sides
        backtracking : :obj:`bool`, optional
            Force backtracking, even if ``tau`` is not equal to ``None``. In this case
            the chosen ``tau`` will be used as the initial guess in the first
            step of backtracking
        beta : :obj:`float`, optional
            Backtracking parameter (must be between 0 and 1)
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 1, 0 excluded).
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        niterback : :obj:`int`, optional
            Max number of iterations of backtracking
        acceleration : :obj:`str`, optional
            Acceleration (``None``, ``vandenberghe`` or ``fista``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            epsg=epsg,
            tau=tau,
            backtracking=backtracking,
            beta=beta,
            eta=eta,
            niter=niter,
            niterback=niterback,
            acceleration=acceleration,
            tol=tol,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(81, show)
        return x, y, self.iiter, self.cost


class AndersonProximalGradient(Solver):
    r"""Proximal gradient with Anderson acceleration

    Solves the following minimization problem using the Proximal
    gradient algorithm with Anderson acceleration:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + \epsilon g(\mathbf{x})

    where :math:`f(\mathbf{x})` is a smooth convex function with a uniquely
    defined gradient and :math:`g(\mathbf{x})` is any convex function that
    has a known proximal operator.

    Notes
    -----
    The Proximal gradient algorithm with Anderson acceleration can be expressed by the
    following recursion [1]_:

    .. math::
        m_k = min(m, k)\\
        \mathbf{g}^{k} = \mathbf{x}^{k} - \tau^k \nabla f(\mathbf{x}^k)\\
        \mathbf{r}^{k} = \mathbf{g}^{k} - \mathbf{g}^{k}\\
        \mathbf{G}^{k} = [\mathbf{g}^{k},..., \mathbf{g}^{k-m_k}]\\
        \mathbf{R}^{k} = [\mathbf{r}^{k},..., \mathbf{r}^{k-m_k}]\\
        \alpha_k = (\mathbf{R}^{kT} \mathbf{R}^{k})^{-1} \mathbf{1} / \mathbf{1}^T
        (\mathbf{R}^{kT} \mathbf{R}^{k})^{-1} \mathbf{1}\\
        \mathbf{y}^{k+1} = \mathbf{G}^{k} \alpha_k\\
        \mathbf{x}^{k+1} = \prox_{\tau^{k+1} g}(\mathbf{y}^{k+1})

    where :math:`m` equals ``nhistory``, :math:`k=1,2,...,n_{iter}`, :math:`\mathbf{y}^{0}=\mathbf{x}^{0}`,
    :math:`\mathbf{y}^{1}=\mathbf{x}^{0} - \tau^0 \nabla f(\mathbf{x}^0)`,
    :math:`\mathbf{x}^{1}=\prox_{\tau^k g}(\mathbf{y}^{1})`, and
    :math:`\mathbf{g}^{0}=\mathbf{y}^{1}`.

    Refer to [1]_ for the guarded version of the algorithm (when ``safeguard=True``).

    .. [1] Mai, V., and Johansson, M. "Anderson Acceleration of Proximal Gradient
       Methods", 2020.

    """

    def _print_setup(self, epsg_print: str, xcomplex: bool = False) -> None:
        self._print_solver(nbar=81)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
        )
        strpar1 = (
            f"tau = {self.tau:4.2e}\t\tepsg = {epsg_print}\t\tniter = {self.niter}"
        )
        strpar2 = f"nhist = {self.nhistory}\t\tepsr = {self.epsr:4.2e}"
        strpar3 = f"guard = {str(self.safeguard)}\t\ttol = {str(self.tol)}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print(strpar3)
        print("-" * 81 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g       J=f+eps*g       tau"
        else:
            head1 = "    Itn              x[0]                  f           g       J=f+eps*g       tau"
        print(head1)

    def _print_step(self, x: NDArray, pg: float | None) -> None:
        if self.tol is None:
            self.pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = self.pf + np.sum(self.epsg[self.iiter - 1] * pg)
        x0 = to_numpy(x[0]) if x.ndim == 1 else to_numpy(x[0, 0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{self.pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}  {self.tau:11.2e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        epsg: float | NDArray = 1.0,
        tau: float | NDArray = 1.0,
        niter: int = 10,
        nhistory: int = 10,
        epsr: float = 1e-10,
        safeguard: bool = False,
        tol: float | None = None,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function (must have ``grad`` implemented)
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor of g function
        tau : :obj:`float` or :obj:`numpy.ndarray`, optional
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. N   ote that :math:`\tau`
            can be chosen to be a vector when dealing with problems with
            multiple right-hand-sides
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        nhistory : :obj:`int`, optional
            Number of previous iterates to be kept in memory (to compute the scaling factors)
        epsr : :obj:`float`, optional
            Scaling factor for regularization added to the inverse of :math:\mathbf{R}^T \mathbf{R}`
        safeguard : :obj:`bool`, optional
            Apply safeguarding strategy to the update (``True``) or not (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess

        """
        self.proxf = proxf
        self.proxg = proxg
        self.x0 = x0
        self.tau = tau
        self.niter = niter
        self.nhistory = nhistory
        self.epsr = epsr
        self.safeguard = safeguard
        self.tol = tol

        self.ncp = get_array_module(x0)

        # check if epgs is a vector
        self.epsg = np.asarray(epsg, dtype=float)
        if self.epsg.size == 1:
            self.epsg = self.epsg * np.ones(niter)
            epsg_print = str(self.epsg[0])
        else:
            epsg_print = "Multi"

        # initialize solver
        y = x0 - self.tau * proxf.grad(x0)
        x = self.proxg.prox(y, self.epsg[0] * self.tau)

        # set history of iterates for Anderson acceleration
        g = y.copy()
        r = g - x0
        self.R, self.G = (
            [
                g,
            ],
            [
                r,
            ],
        )

        # create variables to track the objective function and iterations
        self.pf = self.proxf(x)
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(epsg_print, np.iscomplexobj(x0))
        return x, y

    def step(
        self, x: NDArray, y: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            Andersonproximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            Anderson proximal gradient algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """

        # update fix point
        g = x - self.tau * self.proxf.grad(x)
        r = g - y

        # update history vectors
        self.R.insert(0, r)
        self.G.insert(0, g)
        if self.iiter >= self.nhistory - 1:
            self.R.pop(-1)
            self.G.pop(-1)

        # solve for alpha coefficients
        Rstack = np.vstack(self.R)
        Rinv = np.linalg.pinv(
            Rstack @ Rstack.T + self.epsr * np.linalg.norm(Rstack) ** 2
        )
        ones = np.ones(min(self.nhistory, self.iiter + 2))
        Rinvones = Rinv @ ones
        alpha = Rinvones / (ones[None] @ Rinvones)

        if not self.safeguard:
            # update auxiliary variable
            y = np.vstack(self.G).T @ alpha

            # update main variable
            x = self.proxg.prox(y, self.epsg[self.iiter] * self.tau)
        else:
            # update auxiliary variable
            ytest = np.vstack(self.G).T @ alpha

            # update main variable
            xtest = self.proxg.prox(ytest, self.epsg[self.iiter] * self.tau)

            # check if function is decreased, otherwise do basic PG step
            pfold, self.pf = self.pf, self.proxf(xtest)
            if (
                self.pf
                <= pfold - self.tau * np.linalg.norm(self.proxf.grad(x)) ** 2 / 2
            ):
                y = ytest
                x = xtest
            else:
                x = self.proxg.prox(g, self.epsg[self.iiter] * self.tau)
                y = g

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            self.pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = self.pf + np.sum(self.epsg[self.iiter] * pg)
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pg = 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, y

    def run(
        self,
        x: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the Anderson proximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the Anderson proximal gradient algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, y = self.step(x, y, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        epsg: float | NDArray = 1.0,
        tau: float | NDArray = 1.0,
        niter: int = 10,
        nhistory: int = 10,
        epsr: float = 1e-10,
        safeguard: bool = False,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function (must have ``grad`` implemented)
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor of g function
        tau : :obj:`float` or :obj:`numpy.ndarray`, optional
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. N   ote that :math:`\tau`
            can be chosen to be a vector when dealing with problems with
            multiple right-hand-sides
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        nhistory : :obj:`int`, optional
            Number of previous iterates to be kept in memory (to compute the scaling factors)
        epsr : :obj:`float`, optional
            Scaling factor for regularization added to the inverse of :math:\mathbf{R}^T \mathbf{R}`
        safeguard : :obj:`bool`, optional
            Apply safeguarding strategy to the update (``True``) or not (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            epsg=epsg,
            tau=tau,
            niter=niter,
            nhistory=nhistory,
            epsr=epsr,
            safeguard=safeguard,
            tol=tol,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(81, show)
        return x, y, self.iiter, self.cost


class GeneralizedProximalGradient(Solver):
    r"""Generalized Proximal gradient

    Solves the following minimization problem using Generalized Proximal
    gradient algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^n f_i(\mathbf{x})
        + \sum_{j=1}^m \epsilon_j g_j(\mathbf{x}),~~n,m \in \mathbb{N}^+

    where the :math:`f_i(\mathbf{x})` are smooth convex functions with a uniquely
    defined gradient and the :math:`g_j(\mathbf{x})` are any convex function that
    have a known proximal operator.

    Notes
    -----
    The Generalized Proximal gradient algorithm can be expressed by the
    following recursion [1]_:

    .. math::
        \text{for } j=1,\cdots,n, \\
        ~~~~\mathbf z_j^{k+1} = \mathbf z_j^{k} + \eta
        \left[prox_{\frac{\tau^k \epsilon_j}{w_j} g_j}\left(2 \mathbf{x}^{k} - \mathbf{z}_j^{k}
        - \tau^k \sum_{i=1}^n \nabla f_i(\mathbf{x}^{k})\right) - \mathbf{x}^{k} \right] \\
        \mathbf{x}^{k+1} = \sum_{j=1}^n w_j \mathbf z_j^{k+1} \\

    where :math:`\sum_{j=1}^n w_j=1`. In the current implementation, :math:`w_j=1/n` when
    not provided.

    .. [1] Raguet, H., Fadili, J. and Peyré, G. "Generalized Forward-Backward Splitting",
       arXiv, 2012.

    """

    def _print_setup(self, epsg_print: str, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operators (f): {[type(proxf).__name__ for proxf in self.proxfs]}\n"
            f"Proximal operators (g): {[type(proxg).__name__ for proxg in self.proxgs]}\n"
        )
        strpar1 = f"tau = {self.tau:4.2e}\tepsg = {epsg_print}\tniter = {self.niter}"
        print(strpar)
        print(strpar1)
        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf = np.sum([proxf(x) for proxf in self.proxfs])
            pg = np.sum(
                [
                    eg * proxg(x)
                    for proxg, eg in zip(self.proxgs, self.epsg, strict=True)
                ]
            )
            self.pfg = pf + pg
        x0 = to_numpy(x[0]) if x.ndim == 1 else to_numpy(x[0, 0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxfs: list[ProxOperator],
        proxgs: list[ProxOperator],
        x0: NDArray,
        tau: float,
        epsg: float | NDArray = 1.0,
        weights: NDArray | None = None,
        eta: float = 1.0,
        niter: int = 10,
        acceleration: str | None = None,
        tol: float | None = None,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxfs : :obj:`list`
            Proximal operators of the :math:`f_i` functions (must have ``grad`` implemented)
        proxgs : :obj:`list`
            Proximal operators of the :math:`g_j` functions
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\sum_{i=1}^n \nabla f_i`.
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor(s) of ``g`` function(s)
        weights : :obj:`float`, optional
            Weighting factors of ``g`` functions. Must sum to 1.
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 1, 0 excluded). Note that
            this will be only used when ``acceleration=None``.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        acceleration:  :obj:`str`, optional
            Acceleration (``None``, ``vandenberghe`` or ``fista``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        y : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        """
        self.proxfs = proxfs
        self.proxgs = proxgs
        self.x0 = x0
        self.tau = tau
        self.eta = eta
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # check if weights sum to 1
        self.weights = (
            np.ones(len(proxgs)) / len(proxgs) if weights is None else weights
        )
        if len(self.weights) != len(self.proxgs) or np.sum(self.weights) != 1.0:
            msg = f"weights={self.weights} must be an array of size {len(self.proxgs)} summing to 1"
            raise ValueError(msg)

        # check if epgs is a vector
        self.epsg = np.asarray(epsg, dtype=float)
        if self.epsg.size == 1:
            self.epsg = epsg * np.ones(len(proxgs))
            epsg_print = str(self.epsg[0])
        else:
            epsg_print = "Multi"

        # check acceleration
        if acceleration in [None, "None", "vandenberghe", "fista"]:
            self.acceleration = acceleration
        else:
            msg = "Acceleration should be None, vandenberghe or fista"
            raise NotImplementedError(msg)

        # initialize solver
        x = x0.copy()
        y = x.copy()
        self.zs = [x.copy() for _ in range(len(proxgs))]

        # for accelaration
        self.t = 1.0

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(epsg_print, np.iscomplexobj(x0))
        return x, y

    def step(
        self, x: NDArray, y: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            generalized proximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            generalized proximal gradient algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        xold = x.copy()

        # gradient
        grad = np.zeros_like(x)
        for _, proxf in enumerate(self.proxfs):
            grad += proxf.grad(x)

        # proximal step
        x = np.zeros_like(x)
        for i, proxg in enumerate(self.proxgs):
            ztmp = 2 * y - self.zs[i] - self.tau * grad
            ztmp = proxg.prox(ztmp, self.tau * self.epsg[i] / self.weights[i])
            self.zs[i] += self.eta * (ztmp - y)
            x += self.weights[i] * self.zs[i]

        # update y
        if self.acceleration == "vandenberghe":
            omega = self.iiter / (self.iiter + 3)
        elif self.acceleration == "fista":
            told = self.t
            self.t = (1.0 + np.sqrt(1.0 + 4.0 * self.t**2)) / 2.0
            omega = (told - 1.0) / self.t
        else:
            omega = 0
        y = x + omega * (x - xold)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = np.sum([proxf(x) for proxf in self.proxfs])
            pg = np.sum(
                [
                    eg * proxg(x)
                    for proxg, eg in zip(self.proxgs, self.epsg, strict=True)
                ]
            )
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, y

    def run(
        self,
        x: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the generalized proximal gradient algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the generalized proximal gradient algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, y = self.step(x, y, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

    def solve(  # type: ignore[override]
        self,
        proxfs: list[ProxOperator],
        proxgs: list[ProxOperator],
        x0: NDArray,
        tau: float,
        epsg: float | NDArray = 1.0,
        weights: NDArray | None = None,
        eta: float = 1.0,
        niter: int = 10,
        acceleration: str | None = None,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxfs : :obj:`list`
            Proximal operators of the :math:`f_i` functions (must have ``grad`` implemented)
        proxgs : :obj:`list`
            Proximal operators of the :math:`g_j` functions
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\sum_{i=1}^n \nabla f_i`.
        epsg : :obj:`float` or :obj:`numpy.ndarray`, optional
            Scaling factor(s) of ``g`` function(s)
        weights : :obj:`float`, optional
            Weighting factors of ``g`` functions. Must sum to 1.
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 1, 0 excluded). Note that
            this will be only used when ``acceleration=None``.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        acceleration:  :obj:`str`, optional
            Acceleration (``None``, ``vandenberghe`` or ``fista``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, y = self.setup(
            proxfs=proxfs,
            proxgs=proxgs,
            x0=x0,
            tau=tau,
            epsg=epsg,
            weights=weights,
            eta=eta,
            niter=niter,
            acceleration=acceleration,
            tol=tol,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(65, show)
        return x, y, self.iiter, self.cost


class HQS(Solver):
    r"""Half Quadratic splitting

    Solves the following minimization problem using Half Quadratic splitting
    algorithm:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    Notes
    -----
    The HQS algorithm can be expressed by the following recursion [1]_:

    .. math::

        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k}) \\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k+1})

    for ``gfirst=False``, or

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k}) \\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1})

    for ``gfirst=False``. Note that ``x`` and ``z`` converge to each other,
    however if iterations are stopped too early ``x`` is guaranteed to belong to
    the domain of ``f`` while ``z`` is guaranteed to belong to the domain of ``g``.
    Depending on the problem either of the two may be the best solution.

    .. [1] D., Geman, and C., Yang, "Nonlinear image recovery with halfquadratic
         regularization", IEEE Transactions on Image Processing,
         4, 7, pp. 932-946, 1995.

    """

    def _print_setup(self, tau_print: str, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
        )
        strpar1 = f"tau = {tau_print}\tniter = {self.niter}"
        strpar2 = f"gfirst = {self.gfirst}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float | NDArray,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = True,
        tol: float | None = None,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector (not required when ``gfirst=False``, can pass ``None``)
        tau : :obj:`float` or :obj:`numpy.ndarray`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. Finally note that
            :math:`\tau` can be chosen to be a vector of size ``niter`` such that
            different :math:`\tau` is used at different iterations (i.e., continuation
            strategy)
        z0 : :obj:`numpy.ndarray`, optional
            Initial z vector (not required when ``gfirst=True``)
        niter : :obj:`int`
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        z : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        Raises
        ------
        ValueError
            If both ``x0`` and ``z0`` are set to ``None``

        """
        self.proxf = proxf
        self.proxg = proxg
        self.niter = niter
        self.gfirst = gfirst
        self.tol = tol

        self.ncp = get_array_module(x0)

        # check if tau is a vector
        self.tau = self.ncp.asarray(tau, dtype=float)
        if self.tau.size == 1:
            tau_print = str(np.round(self.tau, 6))
            self.tau = self.tau * np.ones(niter)
        else:
            tau_print = "Variable"

        # initialize solver
        x, z = _x0z0_init(x0, z0)

        # for accelaration
        self.t = 1.0

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(tau_print, np.iscomplexobj(x0))
        return x, z

    def step(
        self, x: NDArray, z: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            HQS algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            HQS algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        z : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        # proximal steps
        if self.gfirst:
            z = self.proxg.prox(x, self.tau[self.iiter])
            x = self.proxf.prox(z, self.tau[self.iiter])
        else:
            x = self.proxf.prox(z, self.tau[self.iiter])
            z = self.proxg.prox(x, self.tau[self.iiter])

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = self.proxf(x)
            pg = self.proxg(x)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, z

    def run(
        self,
        x: NDArray,
        z: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the HQS algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the HQS algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, z = self.step(x, z, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, z

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float | NDArray,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = True,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector (not required when ``gfirst=False``, can pass ``None``)
        tau : :obj:`float` or :obj:`numpy.ndarray`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`. Finally note that
            :math:`\tau` can be chosen to be a vector of size ``niter`` such that
            different :math:`\tau` is used at different iterations (i.e., continuation
            strategy)
        z0 : :obj:`numpy.ndarray`, optional
            Initial z vector (not required when ``gfirst=True``)
        niter : :obj:`int`
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, z = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            tau=tau,
            z0=z0,
            niter=niter,
            gfirst=gfirst,
            tol=tol,
            show=show,
        )

        x, z = self.run(x, z, niter, show=show, itershow=itershow)
        self.finalize(65, show)
        return x, z, self.iiter, self.cost


class ADMM(Solver):
    r"""Alternating Direction Method of Multipliers

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        f(\mathbf{x}) + g(\mathbf{z}) \\
        s.t. \; \mathbf{x}=\mathbf{z}

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{z})` are any convex
    function that has a known proximal operator.

    ADMM can also solve the problem of the form above with a more general
    constraint: :math:`\mathbf{Ax}+\mathbf{Bz}=\mathbf{c}`. This routine implements
    the special case where :math:`\mathbf{A}=\mathbf{I}`, :math:`\mathbf{B}=-\mathbf{I}`,
    and :math:`\mathbf{c}=\mathbf{0}`, as a general algorithm can be obtained for any choice of
    :math:`f` and :math:`g` provided they have a known proximal operator.

    On the other hand, for more general choice of :math:`\mathbf{A}`, :math:`\mathbf{B}`,
    and :math:`\mathbf{c}`, the iterations are not generalizable, i.e. they depend on the choice of
    the :math:`f` and :math:`g` functions. For this reason, we currently only provide an additional
    solver for the special case where :math:`f` is a :class:`pyproximal.proximal.L2`
    operator with a linear operator  :math:`\mathbf{G}` and data :math:`\mathbf{y}`,
    :math:`\mathbf{B}=-\mathbf{I}` and :math:`\mathbf{c}=\mathbf{0}`,
    called :func:`pyproximal.optimization.primal.ADMML2`. Note that for the very same choice
    of :math:`\mathbf{B}` and :math:`\mathbf{c}`, the :func:`pyproximal.optimization.primal.LinearizedADMM`
    can also be used (and this does not require a specific choice of :math:`f`).

    See Also
    --------
    ADMML2: ADMM with L2 misfit function
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    The ADMM algorithm can be expressed by the following recursion [1]_:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{z}^{k} - \mathbf{u}^{k})\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{x}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}

    Note that ``x`` and ``z`` converge to each other, however if iterations are
    stopped too early ``x`` is guaranteed to belong to the domain of ``f``
    while ``z`` is guaranteed to belong to the domain of ``g``. Depending on
    the problem either of the two may be the best solution.

    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. 2011.
        Distributed optimization and statistical learning via the alternating
        direction method of multipliers. Foundations and Trends in Machine
        Learning, 3 (1), 1-122. https://doi.org/10.1561/2200000016.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
        )
        strpar1 = f"tau = {self.tau:6e}\tniter = {self.niter}"
        strpar2 = f"gfirst = {self.gfirst}\t\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = False,
        tol: float | None = None,
        callbackz: bool = False,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector (not required when ``gfirst=False``, can pass ``None``)
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`.
        z0 : :obj:`numpy.ndarray`, optional
            Initial z vector (not required when ``gfirst=True``)
        niter : :obj:`int`
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbackz : :obj:`bool`, optional
            Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        z : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        Raises
        ------
        ValueError
            If both ``x0`` and ``z0`` are set to ``None``

        """
        self.proxf = proxf
        self.proxg = proxg
        self.tau = tau
        self.niter = niter
        self.gfirst = gfirst
        self.tol = tol
        self.callbackz = callbackz

        self.ncp = get_array_module(x0)

        # initialize solver
        x, z = _x0z0_init(x0, z0)
        self.u = self.ncp.zeros_like(x)

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, z

    def step(
        self, x: NDArray, z: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            ADDM algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            ADDM algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        z : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        # proximal steps
        if self.gfirst:
            z = self.proxg.prox(x + self.u, self.tau)
            x = self.proxf.prox(z - self.u, self.tau)
        else:
            x = self.proxf.prox(z - self.u, self.tau)
            z = self.proxg.prox(x + self.u, self.tau)
        self.u = self.u + x - z

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = self.proxf(x)
            pg = self.proxg(x)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, z

    def run(
        self,
        x: NDArray,
        z: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the ADDM algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the ADDM algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, z = self.step(x, z, showstep)
            if self.callbackz:
                self.callback(x, z)
            else:
                self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, z

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = False,
        tol: float | None = None,
        callbackz: bool = False,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector (not required when ``gfirst=False``, can pass ``None``)
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`.
        z0 : :obj:`numpy.ndarray`, optional
            Initial z vector (not required when ``gfirst=True``)
        niter : :obj:`int`
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        callbackz : :obj:`bool`, optional
            Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, z = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            tau=tau,
            z0=z0,
            niter=niter,
            gfirst=gfirst,
            tol=tol,
            callbackz=callbackz,
            show=show,
        )

        x, z = self.run(x, z, niter, show=show, itershow=itershow)
        self.finalize(65, show)
        return x, z, self.iiter, self.cost


class ADMML2(Solver):
    r"""Alternating Direction Method of Multipliers for L2 misfit term

    Solves the following minimization problem using Alternating Direction
    Method of Multipliers:

    .. math::

        \mathbf{x},\mathbf{z}  = \argmin_{\mathbf{x},\mathbf{z}}
        \frac{1}{2}||\mathbf{Op}\mathbf{x} - \mathbf{b}||_2^2 + g(\mathbf{z}) \\
        s.t. \; \mathbf{Ax}=\mathbf{z}

    where :math:`g(\mathbf{z})` is any convex function that has a known proximal operator.

    See Also
    --------
    ADMM: ADMM
    LinearizedADMM: Linearized ADMM

    Notes
    -----
    The ADMM algorithm with L2 misfit term can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = \argmin_{\mathbf{x}} \frac{1}{2}||\mathbf{Op}\mathbf{x}
        - \mathbf{b}||_2^2 + \frac{1}{2\tau} ||\mathbf{Ax} - \mathbf{z}^k + \mathbf{u}^k||_2^2\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{Ax}^{k+1} + \mathbf{u}^{k})\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{Ax}^{k+1} - \mathbf{z}^{k+1}

    .. [1] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. 2011.
        Distributed optimization and statistical learning via the alternating
        direction method of multipliers. Foundations and Trends in Machine
        Learning, 3 (1), 1-122. https://doi.org/10.1561/2200000016.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
            f"Linear operator (Op): {type(self.Op).__name__}\n"
            f"Linear operator (A): {type(self.A).__name__}\n"
        )
        strpar1 = f"tau = {self.tau:6e}\tniter = {self.niter}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(
        self, x: NDArray, Ax: NDArray, pf: float | None, pg: float | None
    ) -> None:
        if self.tol is None:
            pf, pg = (
                0.5 * self.ncp.linalg.norm(self.Op @ x - self.b) ** 2,
                self.proxg(Ax),
            )
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxg: ProxOperator,
        Op: "LinearOperator",
        b: NDArray,
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = False,
        tol: float | None = None,
        callbackz: bool = False,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        Op : :obj:`pylops.LinearOperator`
            Linear operator of data misfit term
        b : :obj:`numpy.ndarray`
            Data
        A : :obj:`pylops.LinearOperator`
            Linear operator of regularization term
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau \in (0, 1/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
        z0 : :obj:`numpy.ndarray`
            Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbackz : :obj:`bool`, optional
            Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        z : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        Raises
        ------
        ValueError
            If both ``x0`` and ``z0`` are set to ``None`` or ``x0`` is set to None

        """
        self.proxg = proxg
        self.Op = Op
        self.b = b
        self.A = A
        self.tau = tau
        self.niter = niter
        self.gfirst = gfirst
        self.tol = tol
        self.callbackz = callbackz

        self.ncp = get_array_module(x0)

        # initialize solver
        x, z = _x0z0_init(x0, z0, A, Opname="A")
        self.u = self.ncp.zeros_like(x)

        # other parameters
        self.sqrttau = 1.0 / sqrt(self.tau)

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, z

    def step(
        self,
        x: NDArray,
        z: NDArray,
        show: bool = False,
        **kwargs_solver: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            ADMML2 algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            ADMML2 algorithm
        show : :obj:`bool`, optional
            Display iteration log
        **kwargs_solver
            Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
            to solve the x-update

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        z : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        if self.gfirst:
            Ax = self.A @ x
            z = self.proxg.prox(Ax + self.u, self.tau)

            # solve augumented system
            x = regularized_inversion(
                self.Op,
                self.b,
                [
                    self.A,
                ],
                x0=x,
                dataregs=[
                    z - self.u,
                ],
                epsRs=[
                    self.sqrttau,
                ],
                **kwargs_solver,
            )[0]
        else:
            # solve augumented system
            x = regularized_inversion(
                self.Op,
                self.b,
                [
                    self.A,
                ],
                x0=x,
                dataregs=[
                    z - self.u,
                ],
                epsRs=[
                    self.sqrttau,
                ],
                **kwargs_solver,
            )[0]
            Ax = self.A @ x
            z = self.proxg.prox(Ax + self.u, self.tau)
        self.u = self.u + Ax - z

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = 0.5 * self.ncp.linalg.norm(self.Op @ x - self.b) ** 2
            pg = self.proxg(Ax)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, Ax, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, z

    def run(
        self,
        x: NDArray,
        z: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
        **kwargs_solver: dict[str, Any],
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the ADDML2 algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the ADDML2 algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        **kwargs_solver
            Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
            to solve the x-update

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, z = self.step(x, z, showstep, **kwargs_solver)
            if self.callbackz:
                self.callback(x, z)
            else:
                self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, z

    def solve(  # type: ignore[override]
        self,
        proxg: ProxOperator,
        Op: "LinearOperator",
        b: NDArray,
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        z0: NDArray | None = None,
        niter: int = 10,
        gfirst: bool = False,
        tol: float | None = None,
        callbackz: bool = False,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
        **kwargs_solver: dict[str, Any],
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        Op : :obj:`pylops.LinearOperator`
            Linear operator of data misfit term
        b : :obj:`numpy.ndarray`
            Data
        A : :obj:`pylops.LinearOperator`
            Linear operator of regularization term
        x0 : :obj:`numpy.ndarray`
            Initial vector (not required when ``gfirst=False``, can pass ``None``)
        tau : :obj:`float`
            Positive scalar weight, which should satisfy the following condition
            to guarantees convergence: :math:`\tau  \in (0, 1/L]` where ``L`` is
            the Lipschitz constant of :math:`\nabla f`.
        z0 : :obj:`numpy.ndarray`, optional
            Initial z vector (not required when ``gfirst=True``)
        niter : :obj:`int`
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping
            criterion). If ``tol=None``, run until ``niter`` is reached
        callbackz : :obj:`bool`, optional
            Modify callback signature to (``callback(x, z)``) when ``callbackz=True``
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.
        **kwargs_solver
            Arbitrary keyword arguments for :py:func:`scipy.sparse.linalg.lsqr` used
            to solve the x-update

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, z = self.setup(
            proxg=proxg,
            Op=Op,
            b=b,
            A=A,
            x0=x0,
            tau=tau,
            z0=z0,
            niter=niter,
            gfirst=gfirst,
            tol=tol,
            callbackz=callbackz,
            show=show,
        )

        x, z = self.run(
            x,
            z,
            niter,
            show=show,
            itershow=itershow,
            **kwargs_solver,
        )
        self.finalize(65, show)
        return x, z, self.iiter, self.cost


class LinearizedADMM(Solver):
    r"""Linearized Alternating Direction Method of Multipliers

    Solves the following minimization problem using Linearized Alternating
    Direction Method of Multipliers:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + g(\mathbf{A}\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    function that has a known proximal operator and :math:`\mathbf{A}` is a
    linear operator.

    See Also
    --------
    ADMM: ADMM
    ADMML2: ADMM with L2 misfit function

    Notes
    -----
    The Linearized-ADMM algorithm can be expressed by the following recursion [1]_:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\mu f}(\mathbf{x}^{k} - \frac{\mu}{\tau}
        \mathbf{A}^H(\mathbf{A} \mathbf{x}^k - \mathbf{z}^k + \mathbf{u}^k))\\
        \mathbf{z}^{k+1} = \prox_{\tau g}(\mathbf{A} \mathbf{x}^{k+1} +
        \mathbf{u}^k)\\
        \mathbf{u}^{k+1} = \mathbf{u}^{k} + \mathbf{A}\mathbf{x}^{k+1} -
        \mathbf{z}^{k+1}

    .. [1] N., Parikh, "Proximal Algorithms", Foundations and Trends
        in Optimization. 2013.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
            f"Linear operator (A): {type(self.A).__name__}\n"
        )
        strpar1 = f"tau = {self.tau:6e}\tmu = {self.mu:6e}"
        strpar2 = f"niter = {self.niter}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(self.Ax)
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        mu: float,
        z0: NDArray | None = None,
        niter: int = 10,
        tol: float | None = None,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`, optional
            Positive scalar weight, which should satisfy the following
            condition to guarantee convergence: :math:`\mu \in (0,
            \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
        mu : :obj:`float`, optional
            Second positive scalar weight, which should satisfy the following
            condition to guarantees convergence: :math:`\mu \in (0,
            \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
        z0 : :obj:`numpy.ndarray`
            Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        z : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        Raises
        ------
        ValueError
            If both ``x0`` and ``z0`` are set to ``None`` or ``x0`` is set to None

        """
        self.proxf = proxf
        self.proxg = proxg
        self.A = A
        self.tau = tau
        self.mu = mu

        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # initialize solver
        x, z = _x0z0_init(x0, z0, A, Opname="A")
        self.Ax = A.matvec(x) if z0 is None else z
        self.u = self.ncp.zeros_like(x)

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, z

    def step(
        self,
        x: NDArray,
        z: NDArray,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            LinearizedADMM algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            LinearizedADMM algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        z : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        x = self.proxf.prox(
            x - self.mu / self.tau * self.A.rmatvec(self.Ax - z + self.u), self.mu
        )

        self.Ax = self.A.matvec(x)
        z = self.proxg.prox(self.Ax + self.u, self.tau)
        self.u = self.u + self.Ax - z

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = self.proxf(x)
            pg = self.proxg(self.Ax)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, z

    def run(
        self,
        x: NDArray,
        z: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the LinearizedADDM algorithm
        z : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the LinearizedADDM algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, z = self.step(x, z, showstep)

            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, z

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        mu: float,
        z0: NDArray | None = None,
        niter: int = 10,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`, optional
            Positive scalar weight, which should satisfy the following
            condition to guarantee convergence: :math:`\mu \in (0,
            \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
        mu : :obj:`float`, optional
            Second positive scalar weight, which should satisfy the following
            condition to guarantees convergence: :math:`\mu \in (0,
            \tau/\lambda_{max}(\mathbf{A}^H\mathbf{A})]`.
        z0 : :obj:`numpy.ndarray`
            Initial auxiliary vector. If ``None``, initialized to ``A @ x0``.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        z : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, z = self.setup(
            proxf=proxf,
            proxg=proxg,
            A=A,
            x0=x0,
            tau=tau,
            mu=mu,
            z0=z0,
            niter=niter,
            tol=tol,
            show=show,
        )

        x, z = self.run(
            x,
            z,
            niter,
            show=show,
            itershow=itershow,
        )
        self.finalize(65, show)
        return x, z, self.iiter, self.cost


class TwIST(Solver):
    r"""Two-step Iterative Shrinkage/Threshold

    Solves the following minimization problem using Two-step Iterative
    Shrinkage/Threshold:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \frac{1}{2}
        ||\mathbf{b} - \mathbf{Ax}||_2^2 + g(\mathbf{x})

    where :math:`\mathbf{A}` is a linear operator and :math:`g(\mathbf{x})`
    is any convex function that has a known proximal operator.

    Notes
    -----
    The TwIST algorithm can be expressed by the following recursion:

    .. math::

        \mathbf{x}^{k+1} = (1-\alpha) \mathbf{x}^{k-1} +
        (\alpha-\beta) \mathbf{x}^k +
        \beta \prox_{g} (\mathbf{x}^k + \mathbf{A}^H
        (\mathbf{b} - \mathbf{A}\mathbf{x}^k)).

    where :math:`\mathbf{x}^{1} = \prox_{g} (\mathbf{x}^0 + \mathbf{A}^T
    (\mathbf{b} - \mathbf{A}\mathbf{x}^0))`.

    The optimal weighting parameters :math:`\alpha` and :math:`\beta` are
    linked to the smallest and largest eigenvalues of
    :math:`\mathbf{A}^H\mathbf{A}` as follows:

    .. math::

        \alpha = 1 + \rho^2 \\
        \beta =\frac{2 \alpha}{\Lambda_{max} + \lambda_{min}}

    where :math:`\rho=\frac{1-\sqrt{k}}{1+\sqrt{k}}` with
    :math:`k=\frac{\lambda_{min}}{\Lambda_{max}}` and
    :math:`\Lambda_{max}=max(1, \lambda_{max})`.

    Experimentally, it has been observed that TwIST is robust to the
    choice of such parameters. Finally, note that in the case of
    :math:`\alpha=1` and :math:`\beta=1`, TwIST is identical to IST.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Linear operator (A): {type(self.A).__name__}\n"
        )
        strpar1 = f"alpha = {self.alpha:6e}\tbeta = {self.beta:6e}"
        strpar2 = f"niter = {self.niter}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxg: ProxOperator,
        A: "LinearOperator",
        b: NDArray,
        x0: NDArray,
        alpha: float | None = None,
        beta: float | None = None,
        eigs: tuple[float, float] | None = None,
        niter: int = 10,
        tol: float | None = None,
        show: bool = False,
    ) -> NDArray:
        r"""Setup solver

        Parameters
        ----------
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator
        b : :obj:`numpy.ndarray`
            Data
        x0 : :obj:`numpy.ndarray`
            Initial vector
        alpha : :obj:`float`, optional
            Positive scalar weight (if ``None``, estimated based on the
            eigenvalues of :math:`\mathbf{A}`, see Notes for details)
        beta : :obj:`float`, optional
            Positive scalar weight (if ``None``, estimated based on the
            eigenvalues of :math:`\mathbf{A}`, see Notes for details)
        eigs : :obj:`tuple`, optional
            Largest and smallest eigenvalues of :math:`\mathbf{A}^H \mathbf{A}`.
            If passed, computes `alpha` and `beta` based on them.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess

        """
        self.proxf = L2(Op=A, b=b)
        self.proxg = proxg
        self.A = A
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # find alpha and beta based on the eigenvalues of A if not provided
        if alpha is None or beta is None:
            if eigs is None:
                emin = A.eigs(neigs=1, which="SM")
                emax = max([1, A.eigs(neigs=1, which="LM")])
            else:
                emax, emin = eigs
            k = emin / emax
            rho = (1 - sqrt(k)) / (1 + sqrt(k))
            self.alpha = 1 + rho**2
            self.beta = 2 * self.alpha / (emax + emin)
        else:
            self.alpha, self.beta = alpha, beta

        # compute proximal of g on initial guess (x_1)
        x = self.proxg.prox(x0 - self.proxf.grad(x0), 1.0)

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x

    def step(
        self,
        x: NDArray,
        xold: NDArray,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            TwisT algorithm
        xold : :obj:`numpy.ndarray`
            Previous model vector
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        xold : :obj:`numpy.ndarray`
            Previous model vector

        """
        # compute new x
        xnew = (
            (1 - self.alpha) * xold
            + (self.alpha - self.beta) * x
            + self.beta * self.proxg.prox(x - self.proxf.grad(x), 1.0)
        )
        # save current x as old (x_i -> x_i-1)
        xold = x.copy()
        # save new x as current (x_i+1 -> x_i)
        x = xnew.copy()

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = self.proxf(x)
            pg = self.proxg(self.Ax)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, xold

    def run(
        self,
        x: NDArray,
        xold: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> NDArray:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the TwisT algorithm
        xold : :obj:`numpy.ndarray`
            Previous estimated model
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, xold = self.step(x, xold, showstep)

            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x

    def solve(  # type: ignore[override]
        self,
        proxg: ProxOperator,
        A: "LinearOperator",
        b: NDArray,
        x0: NDArray,
        alpha: float | None = None,
        beta: float | None = None,
        eigs: tuple[float, float] | None = None,
        niter: int = 10,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator
        b : :obj:`numpy.ndarray`
            Data
        x0 : :obj:`numpy.ndarray`
            Initial vector
        alpha : :obj:`float`, optional
            Positive scalar weight (if ``None``, estimated based on the
            eigenvalues of :math:`\mathbf{A}`, see Notes for details)
        beta : :obj:`float`, optional
            Positive scalar weight (if ``None``, estimated based on the
            eigenvalues of :math:`\mathbf{A}`, see Notes for details)
        eigs : :obj:`tuple`, optional
            Largest and smallest eigenvalues of :math:`\mathbf{A}^H \mathbf{A}`.
            If passed, computes `alpha` and `beta` based on them.
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x = self.setup(
            proxg=proxg,
            A=A,
            b=b,
            x0=x0,
            alpha=alpha,
            beta=beta,
            eigs=eigs,
            niter=niter,
            tol=tol,
            show=show,
        )

        x = self.run(
            x,
            x0,
            niter,
            show=show,
            itershow=itershow,
        )
        self.finalize(65, show)
        return x, self.iiter, self.cost


class DouglasRachfordSplitting(Solver):
    r"""Douglas-Rachford Splitting

    Solves the following minimization problem using Douglas-Rachford Splitting
    algorithm:

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) + g(\mathbf{x})

    where :math:`f(\mathbf{x})` and :math:`g(\mathbf{x})` are any convex
    functions that has known proximal operators.

    Notes
    -----
    The Douglas-Rachford Splitting algorithm can be expressed by the following
    recursion [1]_, [2]_, [3]_, [4]_:

    .. math::

        \mathbf{x}^{k} &= \prox_{\tau g}(\mathbf{y}^k) \\
        \mathbf{y}^{k+1} &= \mathbf{y}^{k} +
        \eta (\prox_{\tau f}(2 \mathbf{x}^{k} - \mathbf{y}^{k})
        - \mathbf{x}^{k})

    .. [1] Patrick L. Combettes and Jean-Christophe Pesquet. 2011. Proximal
        Splitting Methods in Signal Processing. In Fixed-Point Algorithms for
        Inverse Problems in Science and Engineering, Springer, pp. 185-212.
        Algorithm 10.15.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [2] Scott B. Lindstrom and Brailey Sims. 2021. Survey: Sixty Years of
        Douglas-Rachford. Journal of the Australian Mathematical Society, 110,
        3, 333-370. Eq.(15). https://doi.org/10.1017/S1446788719000570
        https://arxiv.org/abs/1809.07181
    .. [3] Ryu, E.K., Yin, W., 2022. Large-Scale Convex Optimization: Algorithms
        & Analyses via Monotone Operators. Cambridge University Press,
        Cambridge. Eq.(2.18). https://doi.org/10.1017/9781009160865
        https://large-scale-book.mathopt.com/
    .. [4] Combettes, P.L., Pesquet, J.-C., 2008. A proximal decomposition
        method for solving convex variational inverse problems. Inverse Problems
        24, 065014. Proposition 3.2. https://doi.org/10.1088/0266-5611/24/6/065014
        https://arxiv.org/abs/0807.2617

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
        )
        strpar1 = f"tau = {self.tau:6e}\teta = {self.eta:6e}\tniter = {self.niter}"
        strpar2 = f"gfirst = {self.gfirst}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]              f           g         J=f+g"
        else:
            head1 = (
                "    Itn              x[0]                  f           g         J=f+g"
            )
        print(head1)

    def _print_step(self, x: NDArray, pf: float | None, pg: float | None) -> None:
        if self.tol is None:
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + pg
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f"{pf:10.3e}  {pg:10.3e}  {self.pfg:10.3e}"
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float,
        eta: float = 1.0,
        niter: int = 10,
        gfirst: bool = True,
        tol: float | None = None,
        callbacky: bool = False,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Positive scalar weight
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 2, 0 excluded).
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbacky : :obj:`bool`, optional
            Modify callback signature to (``callback(x, y)``)
            when ``callbacky=True``
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        y : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable

        """
        self.proxf = proxf
        self.proxg = proxg
        self.tau = tau
        self.eta = eta
        self.niter = niter
        self.gfirst = gfirst
        self.tol = tol
        self.callbacky = callbacky

        self.ncp = get_array_module(x0)

        # initialize solver
        x = x0.copy()
        y = x0.copy()

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, y

    def step(
        self, x: NDArray, y: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            DouglasRachfordSplitting algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            DouglasRachfordSplitting algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        # proximal steps
        if self.gfirst:
            x = self.proxg.prox(y, self.tau)
            y = y + self.eta * (self.proxf.prox(2 * x - y, self.tau) - x)
        else:
            x = self.proxf.prox(y, self.tau)
            y = y + self.eta * (self.proxg.prox(2 * x - y, self.tau) - x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            pf = self.proxf(x)
            pg = self.proxg(x)
            self.pfg = pf + pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, y

    def run(
        self,
        x: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the DouglasRachfordSplitting algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the DouglasRachfordSplitting algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, y = self.step(x, y, showstep)
            if self.callbacky:
                self.callback(x, y)
            else:
                self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

    def solve(  # type: ignore[override]
        self,
        proxf: ProxOperator,
        proxg: ProxOperator,
        x0: NDArray,
        tau: float,
        eta: float = 1.0,
        niter: int = 10,
        gfirst: bool = True,
        tol: float | None = None,
        callbacky: bool = False,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Positive scalar weight
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 2, 0 excluded).
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbacky : :obj:`bool`, optional
            Modify callback signature to (``callback(x, y)``)
            when ``callbacky=True``
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            tau=tau,
            eta=eta,
            niter=niter,
            gfirst=gfirst,
            tol=tol,
            callbacky=callbacky,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(65, show)
        return x, y, self.iiter, self.cost


class PPXA(Solver):
    r"""Parallel Proximal Algorithm (PPXA)

    Solves the following minimization problem using
    Parallel Proximal Algorithm (PPXA):

    .. math::

        \mathbf{x} = \argmin_\mathbf{x} \sum_{i=1}^m f_i(\mathbf{x})

    where :math:`f_i(\mathbf{x})` are any convex
    functions that has known proximal operators.

    See Also
    --------
    ConsensusADMM: Consensus ADMM

    Notes
    -----
    The Parallel Proximal Algorithm (PPXA) can be expressed by the following
    recursion [1]_, [2]_, [3]_, [4]_:

    * :math:`\mathbf{y}_{i}^{0} = \mathbf{x}` or :math:`\mathbf{y}_{i}^{0} = \mathbf{x}_{i}` for :math:`i=1,\ldots,m`
    * :math:`\mathbf{x}^{0} = \sum_{i=1}^m w_i \mathbf{y}_{i}^{0}`
    * for :math:`k = 1, \ldots`

      * for :math:`i = 1, \ldots, m`

        * :math:`\mathbf{p}_{i}^{k} = \prox_{\frac{\tau}{w_i} f_i} (\mathbf{y}_{i}^{k})`

      * :math:`\mathbf{p}^{k} = \sum_{i=1}^{m} w_i \mathbf{p}_{i}^{k}`
      * for :math:`i = 1, \ldots, m`

        * :math:`\mathbf{y}_{i}^{k+1} = \mathbf{y}_{i}^{k} + \eta (2 \mathbf{p}^{k} - \mathbf{x}^{k} - \mathbf{p}_i^{k})`

      * :math:`\mathbf{x}^{k+1} = \mathbf{x}^{k} + \eta (\mathbf{p}^{k} - \mathbf{x}^{k})`

    where :math:`0 < \eta < 2` and
    :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`.
    In the current implementation, :math:`w_i = 1 / m` when not provided.

    References
    ----------
    .. [1] Combettes, P.L., Pesquet, J.-C., 2008. A proximal decomposition
        method for solving convex variational inverse problems. Inverse Problems
        24, 065014. Algorithm 3.1. https://doi.org/10.1088/0266-5611/24/6/065014
        https://arxiv.org/abs/0807.2617
    .. [2] Combettes, P.L., Pesquet, J.-C., 2011. Proximal Splitting Methods in
        Signal Processing, in Fixed-Point Algorithms for Inverse Problems in
        Science and Engineering, Springer, pp. 185-212. Algorithm 10.27.
        https://doi.org/10.1007/978-1-4419-9569-8_10
    .. [3] Bauschke, H.H., Combettes, P.L., 2011. Convex Analysis and Monotone
        Operator Theory in Hilbert Spaces, 1st ed, CMS Books in Mathematics.
        Springer, New York, NY. Proposition 27.8.
        https://doi.org/10.1007/978-1-4419-9467-7
    .. [4] Ryu, E.K., Yin, W., 2022. Large-Scale Convex Optimization: Algorithms
        & Analyses via Monotone Operators. Cambridge University Press,
        Cambridge. Exercise 2.38 https://doi.org/10.1017/9781009160865
        https://large-scale-book.mathopt.com/

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=65)

        strpar = "\n".join(
            [
                f"Proximal operator (f{i}): {type(proxf).__name__}"
                for i, proxf in enumerate(self.proxfs)
            ]
        )
        strpar1 = f"tau = {self.tau:6e}\teta = {self.eta:6e}"
        strpar2 = f"weights = {self.weights}"
        strpar3 = f"niter = {self.niter}\ttol = {self.tol}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print(strpar3)

        print("-" * 65 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]         J=sum_i f_i"
        else:
            head1 = "    Itn              x[0]             J=sum_i f_i"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        if self.tol is None:
            self.pf = self.ncp.sum([self.proxfs[i](x) for i in range(self.nprox)])
            self.pf = self.ncp.sum([self.proxfs[i](x) for i in range(self.nprox)])
        x0 = to_numpy(x[0])
        strx = f"{x0:1.2e}     " if np.iscomplexobj(x) else f"{x0:11.4e}      "
        msg = f"{self.iiter:6g}        " + strx + f"{self.pf:10.3e}"
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxfs: list[ProxOperator],
        x0: NDArray | list[NDArray],
        tau: float,
        eta: float = 1.0,
        weights: NDArray | list[float] | None = None,
        niter: int = 1000,
        tol: float | None = 1e-7,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxfs : :obj:`list`
            A list of proximable functions :math:`f_1, \ldots, f_m`.
        x0 : :obj:`numpy.ndarray` or :obj:`list`
            Initial vector :math:`\mathbf{x}` for all :math:`f_i` if 1-dimensional array
            is provided, or initial vectors :math:`\mathbf{x}_{i}` for each :math:`f_i`
            for :math:`i=1,\ldots,m` if a :obj:`list` of 1-dimensional arrays or a 2-dimensional
            array of size ``(m, d)`` is provided, where ``d`` is the dimension of :math:`\mathbf{x}_{i}`.
        tau : :obj:`float`
            Positive scalar weight
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 2, 0 excluded).
        weights : :obj:`numpy.ndarray` or :obj:`list` or :obj:`None`, optional
            Weights :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`,
            Defaults to None, which means :math:`w_1 = \cdots = w_m = \frac{1}{m}.`
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme.
        tol : :obj:`float`, optional
            Tolerance on change of the solution (used as stopping criterion).
            If ``tol=0``, run until ``niter`` is reached.
        show : :obj:`bool`, optional
            Display iterations log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess
        y : :obj:`numpy.ndarray`
            Initial guess for the auxiliary variable(s)

        """
        self.proxfs = proxfs
        self.tau = tau
        self.eta = eta
        self.weights = weights
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # initialize solver
        self.nprox = len(proxfs)
        if weights is None:
            self.w = self.ncp.full(self.nprox, 1.0 / self.nprox)
        else:
            self.w = self.ncp.asarray(weights)

        if isinstance(x0, list) or x0.ndim == 2:
            y = self.ncp.asarray(x0)  # yi_0 = xi_0, for i = 1, ..., m
        else:
            y = self.ncp.full(
                (self.nprox, x0.size), x0
            )  # y1_0 = y2_0 = ... = ym_0 = x0

        x = self.ncp.mean(y, axis=0)

        # create variables to track the objective function and iterations
        self.pf, self.pfold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, y

    def step(
        self, x: NDArray, y: NDArray, show: bool = False
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            PPXA algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            PPXA algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        x_old = x.copy()

        # proximal steps
        p = self.ncp.stack(
            [self.proxfs[i].prox(y[i], self.tau / self.w[i]) for i in range(self.nprox)]
        )
        pn = self.ncp.sum(self.w[:, None] * p, axis=0)
        y = y + self.eta * (2 * pn - x - p)
        x = x + self.eta * (pn - x)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        # if self.tol is not None:
        #     self.pfold = self.pf
        #     self.pf = self.ncp.sum([self.proxfs[i](x) for i in range(self.nprox)])
        #     if np.abs(1.0 - self.pf / self.pfold) < self.tol:
        #         self.tolbreak = True
        # else:
        #     self.pf = 0.0

        # tolerance check: break iterations if solution does
        # not changeabove tolerance
        if self.tol is not None:
            if self.ncp.abs(x - x_old).max() < self.tol:
                self.tolbreak = True

        self.iiter += 1
        if show:
            self._print_step(x)
        if self.tol is not None or show:
            self.cost.append(float(self.pf))
        return x, y

    def run(
        self,
        x: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the PPXA algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by multiple steps of
            the PPXA algorithm
        niter : :obj:`int`, optional
            Number of iterations. Can be set to ``None`` if already
            provided in the setup call
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model(s)

        """
        niter = self.niter if niter is None else niter
        if niter is None:
            msg = "`niter` must not be None"
            raise ValueError(msg)
        while self.iiter < niter and not self.tolbreak:
            showstep = (
                True
                if show
                and (
                    self.iiter < itershow[0]
                    or niter - self.iiter < itershow[1]
                    or self.iiter % itershow[2] == 0
                )
                else False
            )
            x, y = self.step(x, y, showstep)
            self.callback(x, y)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

    def solve(  # type: ignore[override]
        self,
        proxfs: list[ProxOperator],
        x0: NDArray | list[NDArray],
        tau: float,
        eta: float = 1.0,
        weights: NDArray | list[float] | None = None,
        niter: int = 1000,
        tol: float | None = 1e-7,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxfs : :obj:`list`
            A list of proximable functions :math:`f_1, \ldots, f_m`.
        x0 : :obj:`numpy.ndarray` or :obj:`list`
            Initial vector :math:`\mathbf{x}` for all :math:`f_i` if 1-dimensional array
            is provided, or initial vectors :math:`\mathbf{x}_{i}` for each :math:`f_i`
            for :math:`i=1,\ldots,m` if a :obj:`list` of 1-dimensional arrays or a 2-dimensional
            array of size ``(m, d)`` is provided, where ``d`` is the dimension of :math:`\mathbf{x}_{i}`.
        tau : :obj:`float`
            Positive scalar weight
        eta : :obj:`float`, optional
            Relaxation parameter (must be between 0 and 2, 0 excluded).
        weights : :obj:`numpy.ndarray` or :obj:`list` or :obj:`None`, optional
            Weights :math:`\sum_{i=1}^m w_i = 1, \ 0 < w_i < 1`,
            Defaults to None, which means :math:`w_1 = \cdots = w_m = \frac{1}{m}.`
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme.
        tol : :obj:`float`, optional
            Tolerance on change of the solution (used as stopping criterion).
            If ``tol=0``, run until ``niter`` is reached.
        show : :obj:`bool`, optional
            Display logs
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Additional estimated model(s)
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`list`
            History of the objective function

        """
        x, y = self.setup(
            proxfs=proxfs,
            x0=x0,
            tau=tau,
            eta=eta,
            weights=weights,
            niter=niter,
            tol=tol,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(65, show)
        return x, y, self.iiter, self.cost


# TOD0:
# - make niter optional in setup and fix and prints
#   in _print_setup to not include if not provided
