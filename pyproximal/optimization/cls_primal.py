__all__ = [
    "ProximalPoint",
]

import time
from typing import TYPE_CHECKING, Optional, cast

import numpy as np
from pylops.optimization.callback import _callback_stop
from pylops.utils.backend import get_array_module, to_numpy
from pylops.utils.typing import NDArray, Tmemunit

from pyproximal.optimization.basesolver import Solver
from pyproximal.ProxOperator import ProxOperator
from pyproximal.utils.bilinear import BilinearOperator

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


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

    def memory_usage(
        self,
        show: bool = False,
        unit: Tmemunit = "B",
    ) -> None:
        pass

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
            Number of iterations (default to ``None`` in case a user wants to
            manually step over the solver)
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
        self.pfs: list[float] = []
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
            self.pfs.append(float(self.pf))
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

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        if show:
            self._print_finalize(nbar=60)

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
        pfs : :obj:`numpy.ndarray`
            History of the objective function

        """
        x = self.setup(prox=prox, x0=x0, tau=tau, niter=niter, tol=tol, show=show)
        x = self.run(x, niter, show=show, itershow=itershow)
        self.finalize(show)
        return x, self.iiter, self.pfs


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
        strpar1 = f"tau = {self.tau:4.2e}\t\tbacktrack = {self.backtracking}\tbeta = {self.beta}"
        strpar2 = (
            f"epsg = {epsg_print}\t\tniter = {self.niter}\t\ttol = {str(self.tol)}"
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

    def memory_usage(
        self,
        show: bool = False,
        unit: Tmemunit = "B",
    ) -> None:
        pass

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
            self.epsg = self.epsg * np.ones(niter)
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

        # set initial vectors
        x = x0.copy()
        y = x.copy()

        # for accelaration
        self.t = 1.0

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.pfs: list[float] = []
        self.pfgs: list[float] = []
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
            self.pfgs.append(float(self.pfg))
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

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        if show:
            self._print_finalize(nbar=81)

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
        pfgs : :obj:`numpy.ndarray`
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
        self.finalize(show)
        return x, y, self.iiter, self.pfgs


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

    def memory_usage(
        self,
        show: bool = False,
        unit: Tmemunit = "B",
    ) -> None:
        pass

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

        # set initial vectors
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
        print("Initial pf: %10.3e" % self.pf)
        self.pfg, self.pfgold = np.inf, np.inf
        self.pfs: list[float] = []
        self.pfgs: list[float] = []
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
            self.pfgs.append(float(self.pfg))
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

    def finalize(self, show: bool = False) -> None:
        r"""Finalize solver

        Parameters
        ----------
        show : :obj:`bool`, optional
            Display finalize log

        """
        self.tend = time.time()
        self.telapsed = self.tend - self.tstart
        if show:
            self._print_finalize(nbar=81)

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
        pfgs : :obj:`numpy.ndarray`
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
        self.finalize(show)
        return x, y, self.iiter, self.pfgs
