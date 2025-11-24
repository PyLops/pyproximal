from typing import TYPE_CHECKING, Optional

import numpy as np
from pylops.utils.typing import NDArray, ShapeLike

from pyproximal.ProxOperator import ProxOperator, _check_tau

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class VStack(ProxOperator):
    r"""Vertical stacking.

    Stack a set of N proximal operators vertically. This operator can be used
    for separable inputs, where the overall proximal operator can be computed
    as the stack of proximal operators on parts of the input vector.

    Parameters
    ----------
    ops : :obj:`list`
        Proximal operators to be stacked
    nn : :obj:`list`, optional
        Size of each portion of the input vector (to be used when different
        portions are in consecutive order)
    restr : :obj:`list`, optional
        List of
        :class:`pylops.Restriction` operators extracting the subset of
        interest (to be used when different portions are not in consecutive
        order). It is user responsibility to ensure that all elements of the
        input vector are used exactly once)

    Notes
    -----
    Given an input vector :math:`\mathbf{x}` to which a number of :math:`N`
    functions are applied to different portions of the vector as:

    .. math::
        f(\mathbf{x}) = \sum_{i=1}^N f_i(\mathbf{x}_i)

    the related proximal operator becomes:

    .. math::
        \prox_{\tau f}(\mathbf{x}) = \left(
        \prox_{\tau f_1}(\mathbf{x}_1), \ldots,
        \tau f_N(\mathbf{x}_N) \right)

    """

    def __init__(
        self,
        ops: list["LinearOperator"],
        nn: Optional[list[ShapeLike]] = None,
        restr: Optional[list["LinearOperator"]] = None,
    ) -> None:
        super().__init__(None, any(op.hasgrad for op in ops))
        self.ops = ops

        if nn is not None:
            self.nn = nn
            cum_nn = np.cumsum(nn)
            self.xin = cum_nn[:-1]
            self.xin = np.insert(self.xin, 0, 0)
            self.xend = cum_nn
            # store required size of input
            self.nx = cum_nn[-1]
        elif restr is not None:
            self.restr = restr
            # store required size of input
            self.nx = np.sum([restr.iava.size for restr in self.restr])
        else:
            raise ValueError("Provide either nn or restr")

    def __call__(self, x: NDArray) -> float:
        if x.size != self.nx:
            raise ValueError(
                f"x must have size {self.nx}, instead the provided x has size {x.size}"
            )
        f = 0.0
        if hasattr(self, "nn"):
            for iop, op in enumerate(self.ops):
                f += op(x[self.xin[iop] : self.xend[iop]])
        else:
            for op, restr in zip(self.ops, self.restr):
                f += op(restr.matvec(x))
        return float(f)

    @_check_tau
    def prox(self, x: NDArray, tau: float) -> NDArray:
        if x.size != self.nx:
            raise ValueError(
                f"x must have size {self.nx}, instead the provided x has size {x.size}"
            )
        if hasattr(self, "nn"):
            f = np.hstack(
                [
                    op.prox(x[self.xin[iop] : self.xend[iop]], tau)
                    for iop, op in enumerate(self.ops)
                ]
            )
        else:
            f = np.zeros_like(x)
            for op, restr in zip(self.ops, self.restr):
                f[restr.iava] = op.prox(restr.matvec(x), tau)
        return f

    def grad(self, x: NDArray) -> NDArray:
        if x.size != self.nx:
            raise ValueError(
                f"x must have size {self.nx}, instead the provided x has size {x.size}"
            )
        if hasattr(self, "nn"):
            f = np.hstack(
                [
                    op.grad(x[self.xin[iop] : self.xend[iop]])
                    for iop, op in enumerate(self.ops)
                ]
            )
        else:
            f = np.zeros_like(x)
            for op, restr in zip(self.ops, self.restr):
                f[restr.iava] = op.grad(restr.matvec(x))
        return f
