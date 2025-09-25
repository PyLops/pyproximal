from typing import Any, Callable, List, Sequence, TypeVar

from pylops.utils.backend import get_array_module
from pylops.utils.typing import NDArray

Proj = TypeVar("Proj", bound=Callable[[NDArray], NDArray])
ProxOp = TypeVar("ProxOp", bound=Callable[[NDArray, float], NDArray])
ProxOpOrProj = TypeVar(
    "ProxOpOrProj", Callable[[NDArray], NDArray], Callable[[NDArray, float], NDArray]
)


def _select_impl_by_arity(
    items: Sequence[Any],
    *,
    use_parallel: bool,
    single: ProxOpOrProj,
    two: ProxOpOrProj,
    more: ProxOpOrProj,
) -> ProxOpOrProj:
    """Choose implementation by ``len(items)`` and ``use_parallel``."""
    if not items:
        raise ValueError("items must not be empty")
    if len(items) == 1:
        return single
    if len(items) == 2 and not use_parallel:
        return two
    return more


def dykstra_two(
    x0: NDArray,
    step1: Proj,
    step2: Proj,
    *,
    niter: int,
    tol: float,
) -> NDArray:
    r"""Compute Dykstra's algorithm for :math:`m=2`."""
    ncp = get_array_module(x0)

    x = x0.copy()
    p = ncp.zeros_like(x)
    q = ncp.zeros_like(x)

    for _ in range(niter):
        x_old = x.copy()

        y = step1(x + p)
        p = p + x - y
        x = step2(y + q)
        q = q + y - x

        if max(ncp.abs(x - x_old).max(), ncp.abs(y - x_old).max()) < tol:
            break

    return x


def parallel_dykstra_projection(
    x0: NDArray,
    proj_ops: List[Proj],
    *,
    niter: int,
    tol: float,
) -> NDArray:
    r"""Compute Dykstra's projection algorithm for :math:`m \ge 2`."""
    ncp = get_array_module(x0)

    u = x0.copy()
    m = len(proj_ops)
    z = [ncp.zeros_like(u) for _ in range(m)]

    for _ in range(niter):
        u_old = u.copy()
        u_prev = ncp.array([u.copy() for _ in range(m)])

        for i in range(m):
            u = proj_ops[i](u_prev[i - 1] + z[i])
            z[i] = z[i] + u_prev[i - 1] - u
            u_prev[i] = u

        if max(ncp.abs(u_old - u).max(), ncp.abs(u_prev - u).max()) < tol:
            break

    return u


def parallel_dykstra_prox(
    x0: NDArray,
    prox_ops: List[ProxOp],
    *,
    weights: NDArray | List[float],
    taus: NDArray | List[float],
    niter: int,
    tol: float,
) -> NDArray:
    r"""Compute Dykstra-like proximal algorithm for :math:`m \ge 2`."""
    ncp = get_array_module(x0)

    x = x0.copy()
    m = len(prox_ops)
    z = ncp.stack([x0.copy() for _ in range(m)])
    w = ncp.asarray(weights, dtype=float)
    w /= w.sum()

    for _ in range(niter):
        x_old = x.copy()
        prox_z = ncp.stack([prox_ops[i](z[i], taus[i]) for i in range(m)])
        x = ncp.sum(w[:, None] * prox_z, axis=0)
        z = z + x - prox_z

        if ncp.abs(x - x_old).max() < tol:
            break

    return x
