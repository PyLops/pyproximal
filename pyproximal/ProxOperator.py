from typing import TYPE_CHECKING, Any, Optional, Union
from collections.abc import Callable

import numpy as np
from pylops.utils.typing import NDArray

from pyproximal.utils.backend import cp_dtype

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


def _check_tau(func: Callable[..., NDArray]) -> Callable[..., NDArray]:
    """Check that tau>0

    This utility function is used to decorate every prox and dualprox method
    to check that tau is positive before performing any computation

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if np.any(args[2] <= 0):
            msg = "tau must be positive"
            raise ValueError(msg)
        return func(*args, **kwargs)

    return wrapper


class ProxOperator:
    r"""Common interface for proximal operators of a function.

    This class defines the overarching structure of any proximal operator. It
    contains two main methods, ``prox`` and ``dualprox`` which are both
    implemented by means of the Moreau decomposition assuming explicit
    knowledge of the other method. For this reason any proximal operators that
    subclasses the ``ProxOperator`` class needs at least one of these two
    methods to be implemented directly.

    Moreover, the method ``grad`` is also defined to compute the gradient of
    the Moreau envelope of the function. This function is only called if the
    user does not provide a gradient function when creating the proximal operator.
    The variable ``hasgrad`` is used to indicate if the function has a gradient
    or not (and thus if the ``grad`` method computes the gradient of the actual
    function or of its Moreau envelope).

    .. note:: End users of PyProximal should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyProximal. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator used by the Proximal operator
    hasgrad : :obj:`bool`, optional
        Flag to indicate if the function is differentiable, i.e., has a
        uniquely defined gradient (``True``) or not (``False``).
    sigmame : :obj:`float`, optional
        Relaxation parameter of the Moreau envelope (when ``sigmame`` tends to infinity
        the gradient of the Moreau envelope tends to the gradient of the function itself).
        Refer to the docstring of the ``grad`` method for more details.

    Notes
    -----
    The proximal operator of a function ``f`` is defined as:

    .. math::

        prox_{\tau f} (\mathbf{x}) = \argmin_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    """

    def __init__(
        self,
        Op: Optional["LinearOperator"] = None,
        hasgrad: bool = False,
        sigmame: float = 1.0,
    ) -> None:
        self.Op = Op
        self.hasgrad = hasgrad
        self.sigmame = sigmame

    def __call__(self, x: NDArray) -> bool | float | int:
        """Functional evaluation of the operator.

        Subclasses should implement this. Returns the
        value of the function.
        """
        msg = (
            "This ProxOperator's __call__ method "
            "must be implemented by subclasses to return a float."
        )
        raise NotImplementedError(msg)

    @_check_tau
    def _prox_moreau(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        """Proximal operator applied to a vector via Moreau decomposition"""
        p = x - tau * self.proxdual(x / tau, 1.0 / tau, **kwargs)
        return p

    @_check_tau
    def _proxdual_moreau(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        """Dual proximal operator applied to a vector via Moreau decomposition"""
        pdual = x - tau * self.prox(x / tau, 1.0 / tau, **kwargs)
        return pdual

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        """Proximal operator applied to a vector

        The proximal operator can always be computed given its dual
        proximal operator using the Moreau decomposition as defined in
        :func:`pyproximal.moreau`. For this reason we can easily create a common
        method for all proximal operators that can be evaluated provided the
        dual proximal is implemented.

        However, direct implementations are generally available. This can
        be done by simply implementing ``prox`` for a specific proximal
        operator, which will overwrite the general method.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._prox_moreau(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        """Dual proximal operator applied to a vector

        The dual of a proximal operator can always be computed given its
        proximal operator using the Moreau decomposition as defined in
        :func:`pyproximal.moreau`. For this reason we can easily create a common
        method for all dual proximal operators that can be evaluated provided
        the proximal is implemented.

        However, since the dual of a proximal operator of a function is
        equivalent to the proximal operator of the conjugate function, smarter
        and faster implementation may be available in special cases. This can
        be done by simply implementing ``proxdual`` for a specific proximal
        operator, which will overwrite the general method.

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._proxdual_moreau(x, tau, **kwargs)

    def grad(self, x: NDArray) -> NDArray:
        r"""Gradient of the Moreau envelope of the function.

        This method is only called if the user does not provide a gradient
        because the function is not differentiable. In this case, the gradient
        of the Moreau envelope of the function is computed instead:

        .. math::

            \nabla_\mathbf{x} M_{\sigma f) =
            \frac{1}{sigma} (\mathbf{x} - \prox_{\sigma f}(\mathbf{x}))

        Parameters
        ----------
        x : :obj:`numpy.ndarray`
            Vector

        Returns
        -------
        g : :obj:`numpy.ndarray`
            Gradient vector

        """
        g = (x - self.prox(x, self.sigmame)) / self.sigmame
        return g

    def affine_addition(self, v: NDArray) -> "ProxOperator":
        r"""Affine addition

        Adds the dot-product of vector ``v`` and vector ``x`` (which is passed
        to ``dual`` or ``proxdual``) to the current function.

        This method can also be accessed via the ``+`` operator.

        Parameters
        ----------
        v : :obj:`numpy.ndarray`
            Vector

        Notes
        -----
        The proximal operator of a function :math:`g=f(\mathbf{x}) +
        \mathbf{v}^T \mathbf{x}` is defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) =
            prox_{\tau f} (\mathbf{x} - \tau \mathbf{v})

        """
        if isinstance(v, (np.ndarray, cp_dtype)):
            return _SumOperator(self, v)
        else:
            msg = "v must be a numpy.ndarray or cupy.ndarray"
            raise NotImplementedError(msg)

    def postcomposition(self, sigma: float) -> "ProxOperator":
        r"""Postcomposition

        Multiplies a scalar ``sigma`` to the current function.

        This method can also be accessed via the ``*`` operator.

        Parameters
        ----------
        sigma : :obj:`float`
            Scalar

        Notes
        -----
        The proximal operator of a function :math:`g= \sigma f(\mathbf{x})` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) =
            prox_{\sigma \tau f} (\mathbf{x})

        """
        if isinstance(sigma, float):
            return _PostcompositionOperator(self, sigma)
        else:
            msg = "sigma must be of type float"
            raise NotImplementedError(msg)

    def precomposition(self, a: float, b: float | NDArray) -> "ProxOperator":
        r"""Precomposition

        Multiplies and add scalars ``a`` and ``b`` to ``x`` when evaluating
        the proximal function

        Parameters
        ----------
        a : :obj:`float`
            Multiplicative scalar
        b : :obj:`float` or obj:`numpy.ndarray` or obj:`cupy.ndarray`
            Additive scalar (or vector)

        Notes
        -----
        The proximal operator of a function :math:`g= f(a \mathbf{x} + b)` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) = \frac{1}{a} (
            prox_{a^2 \tau f} (a \mathbf{x} + b) - b)

        """
        if isinstance(a, float) and isinstance(b, (float, np.ndarray, cp_dtype)):  # type: ignore[redundant-expr]
            return _PrecompositionOperator(self, a, b)
        else:
            msg = "a must be of type float and b must be of type float or numpy.ndarray"
            raise NotImplementedError(msg)

    def chain(self, g: "ProxOperator") -> "ProxOperator":
        r"""Chain

        Chains two proximal operators. This must be used with care only when
        aware that the combination of two proximal operators can be simply
        obtained by chaining them

        Parameters
        ----------
        g : :obj:`pyproximal.proximal.ProxOperator`
            Rigth operator

        Notes
        -----
        The proximal operator of the chain of two operators is defined as:

        .. math::

            prox_{\tau f g} (\mathbf{x}) = prox_{\tau g}(prox_{\tau f g}(x))

        """
        return _ChainOperator(self, g)

    def __add__(self, v: NDArray) -> "ProxOperator":
        return self.affine_addition(v)

    def __sub__(self, v: NDArray) -> "ProxOperator":
        return self.__add__(-v)

    def __rmul__(self, sigma: Union[float, int, "ProxOperator"]) -> "ProxOperator":
        if isinstance(sigma, (int, float)):
            return self.postcomposition(sigma)
        else:
            return self.chain(sigma)

    # __rmul__ = __mul__

    def _adjoint(self) -> "_AdjointOperator":
        """Adjoint operator - swaps prox and proxdual"""
        return _AdjointOperator(self)

    H = property(_adjoint)


class _AdjointOperator(ProxOperator):
    def __init__(self, f: "ProxOperator") -> None:
        self.f = f
        super().__init__(None, f.hasgrad)

    def __call__(self, x: NDArray) -> bool | float | int:
        return self.f(x)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.f.proxdual(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.f.prox(x, tau, **kwargs)


class _SumOperator(ProxOperator):
    def __init__(self, f: ProxOperator, v: NDArray) -> None:
        # if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(v, (np.ndarray, cp_dtype)):
            msg = "Second input must be a numpy.ndarray or cupy.ndarray"
            raise ValueError(msg)
        self.f, self.v = f, v
        super().__init__(None, f.hasgrad)

    def __call__(self, x: NDArray) -> float:
        f: float = self.f(x) + np.dot(self.v, x)
        return f

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.f.prox(x - tau * self.v, tau)

    def grad(self, x: NDArray) -> NDArray:
        return self.f.grad(x) + self.v


class _ChainOperator(ProxOperator):
    def __init__(self, f: ProxOperator, g: ProxOperator) -> None:
        # if not isinstance(f, ProxOperator) or not isinstance(g, ProxOperator):
        #    raise ValueError('Inputs must be a ProxOperator')
        self.f, self.g = f, g
        super().__init__(None, f.hasgrad and g.hasgrad)

    def __call__(self, x: NDArray) -> bool | float | int:
        return self.g(self.f(x))

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.g.prox(self.f.prox(x, tau), tau)

    def grad(self, x: NDArray) -> NDArray:
        pass


class _PostcompositionOperator(ProxOperator):
    def __init__(self, f: ProxOperator, sigma: float) -> None:
        # if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(sigma, float):
            msg = "Second input must be a float"
            raise ValueError(msg)
        self.f, self.sigma = f, sigma
        super().__init__(None, f.hasgrad)

    def __call__(self, x: NDArray) -> bool | float | int:
        return self.sigma * self.f(x)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return self.f.prox(x, self.sigma * tau)

    def grad(self, x: NDArray) -> NDArray:
        return self.sigma * self.f.grad(x)


class _PrecompositionOperator(ProxOperator):
    def __init__(self, f: ProxOperator, a: float, b: float | NDArray) -> None:
        # if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(a, float):
            msg = "Second input must be a float"
            raise ValueError(msg)
        if not isinstance(b, (float, np.ndarray, cp_dtype)):
            msg = "Third input must be a float, numpy.ndarray, or cupy.ndarray"
            raise ValueError(msg)
        self.f, self.a, self.b = f, a, b
        super().__init__(None, f.hasgrad)

    def __call__(self, x: NDArray) -> NDArray:
        return self.f(self.a * x + self.b)

    @_check_tau
    def prox(self, x: NDArray, tau: float, **kwargs: Any) -> NDArray:
        return (self.f.prox(self.a * x + self.b, (self.a**2) * tau) - self.b) / self.a

    def grad(self, x: NDArray) -> NDArray:
        return self.a * self.f.grad(self.a * x + self.b)
