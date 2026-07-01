__all__ = [
    "PrimalDual",
    "AdaptivePrimalDual",
]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pylops
from pylops.optimization.callback import Callbacks
from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray

from pyproximal.optimization.basesolver import Solver
from pyproximal.ProxOperator import ProxOperator

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


class PrimalDual(Solver):
    r"""Primal-dual algorithm

    Solves the following (possibly) nonlinear minimization problem using
    the general version of the first-order primal-dual algorithm of [1]_:

    .. math::

        \min_{\mathbf{x} \in X} g(\mathbf{Ax}) + f(\mathbf{x}) +
        \mathbf{z}^T \mathbf{x}

    where :math:`\mathbf{A}` is a linear operator, :math:`f`
    and :math:`g` can be any convex functions that have a known proximal
    operator.

    This functional is effectively minimized by solving its equivalent
    primal-dual problem (primal in :math:`f`, dual in :math:`g`):

    .. math::

        \min_{\mathbf{x} \in X} \max_{\mathbf{y} \in Y}
        \mathbf{y}^T(\mathbf{Ax}) + \mathbf{z}^T \mathbf{x} +
        f(\mathbf{x}) - g^*(\mathbf{y})

    where :math:`\mathbf{y}` is the so-called dual variable.

    Notes
    -----
    The Primal-dual algorithm can be expressed by the following recursion
    (``gfirst=True``):

    .. math::

        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k})\\
        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k+1} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k)

    where :math:`\tau \mu \lambda_{max}(\mathbf{A}^H\mathbf{A}) < 1`.

    Alternatively for ``gfirst=False`` the scheme becomes:

    .. math::

        \mathbf{x}^{k+1} = \prox_{\tau f}(\mathbf{x}^{k} -
        \tau (\mathbf{A}^H \mathbf{y}^{k} + \mathbf{z})) \\
        \bar{\mathbf{x}}^{k+1} = \mathbf{x}^{k+1} +
        \theta (\mathbf{x}^{k+1} - \mathbf{x}^k) \\
        \mathbf{y}^{k+1} = \prox_{\mu g^*}(\mathbf{y}^{k} +
        \mu \mathbf{A}\bar{\mathbf{x}}^{k+1})

    .. [1] A., Chambolle, and T., Pock, "A first-order primal-dual algorithm for
        convex problems with applications to imaging", Journal of Mathematical
        Imaging and Vision, 40, 8pp. 120-145. 2011.

    """

    def _print_setup(
        self, tau_print: str, mu_print: str, xcomplex: bool = False
    ) -> None:
        self._print_solver(nbar=85)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
            f"Linear operator (A): {type(self.A).__name__}\n"
            f"Additional vector (z): {None if self.z is None else 'vector'}\n"
        )
        strpar1 = f"tau = {tau_print}\tmu = {mu_print}\ttheta = {self.theta:6.2e}"
        strpar2 = f"tol = {str(self.tol)}\tniter = {self.niter}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print("-" * 85 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]                 f            g          z^x       J=f+g+z^x"
        else:
            head1 = "    Itn              x[0]                      f          g           z^x        J=f+g+z^x"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        if self.tol is None:
            self.pf = self.proxf(x)
            self.pg = self.proxg(self.A.matvec(x))
            self.zx = 0.0 if self.z is None else self.ncp.dot(self.z, x)
        pf = 0.0 if isinstance(self.pf, bool) else self.pf
        pg = 0.0 if isinstance(self.pg, bool) else self.pg
        self.pfg = pf + pg + self.zx
        strx = f"{x[0]:1.2e}        " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f" {self.pf:11.4e} "
            + f"{self.pg:11.4e} "
            + f" {self.zx:11.4e} "
            + f" {self.pfg:11.4e} "
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: "ProxOperator",
        proxg: "ProxOperator",
        A: "LinearOperator",
        x0: NDArray,
        tau: float | NDArray,
        mu: float | NDArray,
        y0: NDArray | None = None,
        z: NDArray | None = None,
        theta: float = 1.0,
        gfirst: bool = True,
        niter: int | None = None,
        tol: float | None = None,
        callbacky: bool = False,
        show: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        r"""Setup solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator of g
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float` or :obj:`numpy.ndarray`
            Stepsize of subgradient of :math:`f`. This can be constant
            or function of iterations (in the latter cases provided as np.ndarray)
        mu : :obj:`float` or :obj:`numpy.ndarray`
            Stepsize of subgradient of :math:`g^*`. This can be constant
            or function of iterations (in the latter cases provided as np.ndarray)
        y0 : :obj:`numpy.ndarray`
            Initial auxiliary vector. If ``None``, set to zero
        z : :obj:`numpy.ndarray`, optional
            Additional vector
        theta : :obj:`float`
            Scalar between 0 and 1 that defines the update of the
            :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
            special case that represents the semi-implicit classical Arrow-Hurwicz
            algorithm
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme (default to ``None``
            in case a user wants to manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbacky : :obj:`bool`, optional
            Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess of model vector
        xhat : :obj:`numpy.ndarray`
            Initial guess of extrapolated model vector
        y : :obj:`numpy.ndarray`
            Initial guess of additional model vector

        """
        self.proxf = proxf
        self.proxg = proxg
        self.A = A
        self.z = z
        self.theta = theta
        self.gfirst = gfirst
        self.niter = niter
        self.tol = tol
        self.callbacky = callbacky

        self.ncp = get_array_module(x0)

        # check if tau is a vector
        self.tau = self.ncp.asarray(tau, dtype=np.float32)
        if self.tau.size == 1:
            if niter is None:
                tau_print = str(np.round(self.tau, 6))
            else:
                self.tau = self.tau * self.ncp.ones(niter, dtype=np.float32)
                tau_print = str(np.round(self.tau[0], 6))
        else:
            tau_print = "Variable"

        # check if tau is a vector
        self.mu = self.ncp.asarray(mu, dtype=np.float32)
        if self.mu.size == 1:
            if niter is None:
                mu_print = str(np.round(self.mu, 6))
            else:
                self.mu = self.mu * self.ncp.ones(niter, dtype=np.float32)
                mu_print = str(np.round(self.mu[0], 6))
        else:
            mu_print = "Variable"

        # initialize solver
        x = x0.copy()
        y = (
            y0.copy()
            if y0 is not None
            else self.ncp.zeros(self.A.shape[0], dtype=x.dtype)
        )
        xhat = x.copy()

        # create variables to track the objective function and iterations
        pf = self.proxf(x)
        pg = self.proxg(self.A.matvec(x))
        zx = 0.0 if z is None else self.ncp.dot(self.z, x)
        pfg = pf + pg + zx
        self.pfg, self.pfgold = pfg, pfg
        self.cost: list[float] = []
        self.cost.append(float(self.pfg))
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(tau_print, mu_print, np.iscomplexobj(x0))
        return x, xhat, y

    def step(
        self,
        x: NDArray,
        xhat: NDArray,
        y: NDArray,
        show: bool = False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            Primal-dual algorithm
        xhat : :obj:`numpy.ndarray`
            Current extrapolated model vector to be updated by a step of the
            Primal-dual algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            Primal-dual algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        xhat : :obj:`numpy.ndarray`
            Updated extrapolated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        xold = x.copy()

        # define tau for current iteration
        if self.tau.ndim == 0:
            tau = self.tau
        else:
            tau = self.tau[self.iiter]

        # define mu for current iteration
        if self.mu.ndim == 0:
            mu = self.mu
        else:
            mu = self.mu[self.iiter]

        if self.gfirst:
            y = self.proxg.proxdual(y + mu * self.A.matvec(xhat), mu)
            ATy = self.A.rmatvec(y)
            if self.z is not None:
                ATy += self.z
            x = self.proxf.prox(x - tau * ATy, tau)
            xhat = x + self.theta * (x - xold)
        else:
            ATy = self.A.rmatvec(y)
            if self.z is not None:
                ATy += self.z
            x = self.proxf.prox(x - tau * ATy, tau)
            xhat = x + self.theta * (x - xold)
            y = self.proxg.proxdual(y + mu * self.A.matvec(xhat), mu)

        # tolerance check: break iterations if overall
        # objective does not decrease below tolerance
        if self.tol is not None:
            self.pfgold = self.pfg
            self.pf = self.proxf(x)
            self.pg = self.proxg(self.A.matvec(x))
            self.zx = 0.0 if self.z is None else self.ncp.dot(self.z, x)
            self.pfg = self.pf + self.pg + self.zx
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True

        self.iiter += 1
        if show:
            self._print_step(x)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, xhat, y

    def run(
        self,
        x: NDArray,
        xhat: NDArray,
        y: NDArray,
        niter: int | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, NDArray]:
        r"""Run solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by multiple steps of
            the Primal-dual algorithm
        xhat : :obj:`numpy.ndarray`
            Current extrapolated model vector to be updated by multiple steps of
            the Primal-dual algorithm
        y : :obj:`numpy.ndarray`
            Current additional model vector to be updated by multiple steps of
            the Primal-dual algorithm
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
        xhat : :obj:`numpy.ndarray`
            Estimated extrapolated model
        y : :obj:`numpy.ndarray`
            Estimated additional model

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
            x, xhat, y = self.step(x, xhat, y, showstep)
            if self.callbacky:
                self.callback(x, y)
            else:
                self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, xhat, y

    def solve(  # type: ignore[override]
        self,
        proxf: "ProxOperator",
        proxg: "ProxOperator",
        A: "LinearOperator",
        x0: NDArray,
        tau: float | NDArray,
        mu: float | NDArray,
        y0: NDArray | None = None,
        z: NDArray | None = None,
        theta: float = 1.0,
        gfirst: bool = True,
        niter: int = 10,
        tol: float | None = None,
        callbacky: bool = False,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, NDArray, int, NDArray]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator of g
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float` or :obj:`numpy.ndarray`
            Stepsize of subgradient of :math:`f`. This can be constant
            or function of iterations (in the latter cases provided as np.ndarray)
        mu : :obj:`float` or :obj:`numpy.ndarray`
            Stepsize of subgradient of :math:`g^*`. This can be constant
            or function of iterations (in the latter cases provided as np.ndarray)
        y0 : :obj:`numpy.ndarray`
            Initial auxiliary vector. If ``None``, set to zero
        z : :obj:`numpy.ndarray`, optional
            Additional vector
        theta : :obj:`float`
            Scalar between 0 and 1 that defines the update of the
            :math:`\bar{\mathbf{x}}` variable - note that ``theta=0`` is a
            special case that represents the semi-implicit classical Arrow-Hurwicz
            algorithm
        gfirst : :obj:`bool`, optional
            Apply Proximal of operator ``g`` first (``True``) or Proximal of
            operator ``f`` first (``False``)
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme
        tol : :obj:`float`, optional
            Tolerance on change of objective function (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        callbacky : :obj:`bool`, optional
            Modify callback signature to (``callback(x, y)``) when ``callbacky=True``
        show : :obj:`bool`, optional
            Display setup log
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        xhat : :obj:`numpy.ndarray`
            Estimated extrapolated model
        y : :obj:`numpy.ndarray`
            Estimated additional model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`numpy.ndarray`
            History of the objective function

        """
        x, xhat, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            A=A,
            x0=x0,
            tau=tau,
            mu=mu,
            y0=y0,
            z=z,
            theta=theta,
            gfirst=gfirst,
            niter=niter,
            tol=tol,
            callbacky=callbacky,
            show=show,
        )
        x, xhat, y = self.run(x, xhat, y, niter, show=show, itershow=itershow)
        self.finalize(85, show)
        return x, xhat, y, self.iiter, self.cost


class AdaptivePrimalDual(Solver):
    r"""Adaptive Primal-dual algorithm

    Solves the minimization problem in
    :func:`pyproximal.optimization.primaldual.PrimalDual`
    using an adaptive version of the first-order primal-dual algorithm of [1]_.
    The main advantage of this method is that step sizes :math:`\tau` and
    :math:`\mu` are changing through iterations, improving the overall speed
    of convergence of the algorithm.

    Notes
    -----
    The Adative Primal-dual algorithm shares the the same iterations of the
    original :func:`pyproximal.optimization.cls_primaldual.PrimalDual` solver.
    The main difference lies in the fact that the step sizes ``tau`` and ``mu``
    are adaptively changed at each iteration leading to faster converge.

    Changes are applied by tracking the norm of the primal and dual
    residuals. When their mutual ratio increases beyond a certain treshold
    ``delta`` the step lenghts are updated to balance the minimization and
    maximization part of the overall optimization process.

    .. [1] T., Goldstein, M., Li, X., Yuan, E., Esser, R., Baraniuk, "Adaptive
        Primal-Dual Hybrid Gradient Methods for Saddle-Point Problems",
        ArXiv, 2013.

    """

    def _print_setup(self, xcomplex: bool = False) -> None:
        self._print_solver(nbar=85)

        strpar = (
            f"Proximal operator (f): {type(self.proxf).__name__}\n"
            f"Proximal operator (g): {type(self.proxg).__name__}\n"
            f"Linear operator (A): {type(self.A).__name__}\n"
            f"Additional vector (z): {None if self.z is None else 'vector'}\n"
        )
        strpar1 = f"tau0 = {self.tau:6.2e}\tmu0 = {self.mu:6.2e}"
        strpar2 = f"alpha0 = {str(self.alpha)}\teta0 = {self.niter}"
        strpar3 = f"s = {str(self.s)}\t\tdelta = {self.delta}"
        strpar4 = f"tol = {str(self.tol)}\tniter = {self.niter}"
        print(strpar)
        print(strpar1)
        print(strpar2)
        print(strpar3)
        print(strpar4)
        print("-" * 85 + "\n")
        if not xcomplex:
            head1 = "    Itn           x[0]                 f            g          z^x       J=f+g+z^x"
        else:
            head1 = "    Itn              x[0]                     f         g          z^x       J=f+g+z^x"
        print(head1)

    def _print_step(self, x: NDArray) -> None:
        self.pf = self.proxf(x)
        self.pg = self.proxg(self.A.matvec(x))
        self.zx = 0.0 if self.z is None else self.ncp.dot(self.z, x)
        pf = 0.0 if isinstance(self.pf, bool) else self.pf
        pg = 0.0 if isinstance(self.pg, bool) else self.pg
        self.pfg = pf + pg + self.zx
        strx = f"{x[0]:1.2e}        " if np.iscomplexobj(x) else f"{x[0]:11.4e}        "
        msg = (
            f"{self.iiter:6g}        "
            + strx
            + f" {self.pf:11.4e} "
            + f"{self.pg:11.4e} "
            + f" {self.zx:11.4e} "
            + f" {self.pfg:11.4e} "
        )
        print(msg)

    def setup(  # type: ignore[override]
        self,
        proxf: "ProxOperator",
        proxg: "ProxOperator",
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        mu: float,
        alpha: float = 0.5,
        eta: float = 0.95,
        s: float = 1.0,
        delta: float = 1.5,
        z: NDArray | None = None,
        niter: int | None = None,
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
            Linear operator of g
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Stepsize of subgradient of :math:`f`
        mu : :obj:`float`
            Stepsize of subgradient of :math:`g^*`
        alpha : :obj:`float`, optional
            Initial adaptivity level (must be between 0 and 1)
        eta : :obj:`float`, optional
            Scaling of adaptivity level to be multipled to the current alpha every
            time the norm of the two residuals start to diverge (must be between
            0 and 1)
        s : :obj:`float`, optional
            Scaling of residual balancing principle
        delta : :obj:`float`, optional
            Balancing factor. Step sizes are updated only when their ratio exceeds
            this value.
        z : :obj:`numpy.ndarray`, optional
            Additional vector
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme (default to ``None``
            in case a user wants to manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on x/y updates (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display setup log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Initial guess

        """
        self.proxf = proxf
        self.proxg = proxg
        self.A = A
        self.tau = tau
        self.mu = mu
        self.alpha = alpha
        self.eta = eta
        self.s = s
        self.delta = delta
        self.z = z
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # initialize solver
        x = x0.copy()
        y = self.ncp.zeros(self.A.shape[0], dtype=x.dtype)

        # initialize additional variables
        self.Ax = self.ncp.zeros(self.A.shape[0], dtype=x.dtype)
        self.ATy = self.ncp.zeros(self.A.shape[1], dtype=x.dtype)

        # initialize arrays to store history of parameters
        self.taus = []
        self.mus = []
        self.alphas = []
        self.taus.append(tau)
        self.mus.append(mu)
        self.alphas.append(alpha)
        if self.tol is None:
            self.p = self.d = None
        else:
            self.p = self.d = 0.0

        # create variables to track the objective function and iterations
        self.pf = self.proxf(x)
        self.pg = self.proxg(self.A.matvec(x))
        self.zx = 0.0 if self.z is None else self.ncp.dot(self.z, x)
        pfg = self.pf + self.pg + self.zx
        self.pfg, self.pfgold = pfg, pfg
        self.cost: list[float] = []
        self.cost.append(float(self.pfg))
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(np.iscomplexobj(x0))
        return x, y

    def step(
        self,
        x: NDArray,
        y: NDArray,
        show: bool = False,
    ) -> tuple[NDArray, NDArray]:
        r"""Run one step of solver

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Current model vector to be updated by a step of the
            Adaptive Primal-dual algorithm
        y : :obj:`numpy.ndarray`
            Additional model vector to be updated by a step of the
            Adaptive Primal-dual algorithm
        show : :obj:`bool`, optional
            Display iteration log

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Updated model vector
        y : :obj:`numpy.ndarray`
            Updated additional model vector

        """
        # store old values
        xold = x.copy()
        yold = y.copy()
        Axold = self.Ax.copy()
        ATyold = self.ATy.copy()

        # proxf
        if self.z is not None:
            self.ATy += self.z
        x = self.proxf.prox(x - self.tau * self.ATy, self.tau)
        self.Ax = self.A.matvec(x)
        Axhat = 2 * self.Ax - Axold

        # proxg
        y = self.proxg.proxdual(y + self.mu * Axhat, self.mu)
        self.ATy = self.A.rmatvec(y)

        # update steps
        if self.z is not None:
            self.p = float(
                np.linalg.norm(
                    (xold - x) / self.tau
                    - (ATyold - self.ATy)
                    - self.A.rmatvec(self.z)
                    + self.z
                )
            )
        else:
            self.p = float(np.linalg.norm((xold - x) / self.tau - (ATyold - self.ATy)))
        self.d = float(np.linalg.norm((yold - y) / self.mu - (Axold - self.Ax)))

        if self.p > self.s * self.d * self.delta:
            self.tau /= 1 - self.alpha
            self.mu *= 1 - self.alpha
            self.alpha *= self.eta
        elif self.p < self.s * self.d / self.delta:
            self.tau *= 1 - self.alpha
            self.mu /= 1 - self.alpha
            self.alpha *= self.eta

        # save history of steps
        self.taus.append(self.tau)
        self.mus.append(self.mu)
        self.alphas.append(self.alpha)

        # tolerance check: break iterations if
        # x/y updates do not decrease
        # below tolerance
        if self.p <= self.tol or self.d <= self.tol:
            self.tolbreak = True

        self.iiter += 1
        if show:
            self._print_step(x)
        if show:
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
            the Adaptive Primal-dual algorithm
        y : :obj:`numpy.ndarray`
            Current additional model vector to be updated by multiple steps of
            the Adaptive Primal-dual algorithm
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
            Estimated additional model

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
        proxf: "ProxOperator",
        proxg: "ProxOperator",
        A: "LinearOperator",
        x0: NDArray,
        tau: float,
        mu: float,
        alpha: float = 0.5,
        eta: float = 0.95,
        s: float = 1.0,
        delta: float = 1.5,
        z: NDArray | None = None,
        niter: int | None = None,
        tol: float | None = None,
        show: bool = False,
        itershow: tuple[int, int, int] = (10, 10, 10),
    ) -> tuple[NDArray, NDArray, int, NDArray, tuple[NDArray, NDArray, NDArray]]:
        r"""Run entire solver

        Parameters
        ----------
        proxf : :obj:`pyproximal.ProxOperator`
            Proximal operator of f function
        proxg : :obj:`pyproximal.ProxOperator`
            Proximal operator of g function
        A : :obj:`pylops.LinearOperator`
            Linear operator of g
        x0 : :obj:`numpy.ndarray`
            Initial vector
        tau : :obj:`float`
            Stepsize of subgradient of :math:`f`
        mu : :obj:`float`
            Stepsize of subgradient of :math:`g^*`
        alpha : :obj:`float`, optional
            Initial adaptivity level (must be between 0 and 1)
        eta : :obj:`float`, optional
            Scaling of adaptivity level to be multipled to the current alpha every
            time the norm of the two residuals start to diverge (must be between
            0 and 1)
        s : :obj:`float`, optional
            Scaling of residual balancing principle
        delta : :obj:`float`, optional
            Balancing factor. Step sizes are updated only when their ratio exceeds
            this value.
        z : :obj:`numpy.ndarray`, optional
            Additional vector
        niter : :obj:`int`, optional
            Number of iterations of iterative scheme (default to ``None``
            in case a user wants to manually step over the solver)
        tol : :obj:`float`, optional
            Tolerance on x/y updates (used as stopping criterion). If
            ``tol=None``, run until ``niter`` is reached
        show : :obj:`bool`, optional
            Display setup log
        itershow : :obj:`tuple`, optional
            Display set log for the first N1 steps, last N2 steps,
            and every N3 steps in between where N1, N2, N3 are the
            three element of the list.

        Returns
        -------
        x : :obj:`numpy.ndarray`
            Estimated model
        y : :obj:`numpy.ndarray`
            Estimated additional model
        iiter : :obj:`int`
            Number of executed iterations
        cost : :obj:`numpy.ndarray`
            History of the objective function
        steps : :obj:`tuple`
            Tau, mu and alpha evolution through iterations

        """
        x, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            A=A,
            x0=x0,
            tau=tau,
            mu=mu,
            alpha=alpha,
            eta=eta,
            s=s,
            delta=delta,
            z=z,
            niter=niter,
            tol=tol,
            show=show,
        )
        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        steps = (
            np.array(self.taus),
            np.array(self.mus),
            np.array(self.alphas),
        )

        self.finalize(85, show)
        return x, y, self.iiter, self.cost, steps
