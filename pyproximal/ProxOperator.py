import numpy as np


def _check_tau(func):
    """Check that tau>0

    This utility function is used to decorate every prox and dualprox method
    to check that tau is positive before performing any computation

    """
    def wrapper(*args, **kwargs):
        if np.any(args[2] <= 0):
            raise ValueError('tau must be positive')
        return func(*args, **kwargs)
    return wrapper


class ProxOperator(object):
    r"""Common interface for proximal operators of a function.

    This class defines the overarching structure of any proximal operator. It
    contains two main methods, ``prox`` and ``dualprox`` which are both
    implemented by means of the Moreau decomposition assuming explicit
    knowledge of the other method. For this reason any proximal operators that
    subclasses the ``ProxOperator`` class needs at least one of these two
    methods to be implemented directly.

    .. note:: End users of PyProx should not use this class directly but simply
      use operators that are already implemented. This class is meant for
      developers and it has to be used as the parent class of any new operator
      developed within PyProx. Find more details regarding implementation of
      new operators at :ref:`addingoperator`.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator used by the Proximal operator
    hasgrad : :obj:`bool`, optional
        Flag to indicate if the function is differentiable, i.e., has a
        uniquely defined gradient (``True``) or not (``False``).

    Notes
    -----
    The proximal operator of a function ``f`` is defined as:

    .. math::

        prox_{\tau f} (\mathbf{x}) = \argmin_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}||\mathbf{y} - \mathbf{x}||^2_2

    """
    def __init__(self, Op=None, hasgrad=False):
        self.Op = Op
        self.hasgrad = hasgrad

    @_check_tau
    def _prox_moreau(self, x, tau, **kwargs):
        """Proximal operator applied to a vector via Moreau decomposition

        """
        p = x - tau * self.proxdual(x / tau, 1. / tau, **kwargs)
        return p

    @_check_tau
    def _proxdual_moreau(self, x, tau, **kwargs):
        """Dual proximal operator applied to a vector via Moreau decomposition

        """
        pdual = x - tau * self.prox(x / tau, 1. / tau, **kwargs)
        return pdual

    @_check_tau
    def prox(self, x, tau, **kwargs):
        """Proximal operator applied to a vector

        The  proximal operator can always be computed given its dual
        proximal operator using the Moreau decomposition as defined in
        :func:`pyprox.moreau`. For this reason we can easily create a common
        method for all proximal operators that can be evaluated provided the
        dual proximal is implemented.

        However, direct implementations are generally available. This can
        be done by simply implementing ``prox`` for a specific proximal
        operator, which will overwrite the general method.


        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._prox_moreau(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x, tau, **kwargs):
        """Dual proximal operator applied to a vector

        The dual of a proximal operator can always be computed given its
        proximal operator using the Moreau decomposition as defined in
        :func:`pyprox.moreau`. For this reason we can easily create a common
        method for all dual proximal operators that can be evaluated provided
        he proximal is implemented.

        However, since the dual of a proximal operator of a function is
        equivalent to the proximal operator of the conjugate function, smarter
        and faster implementation may be available in special cases. This can
        be done by simply implementing ``proxdual`` for a specific proximal
        operator, which will overwrite the general method.

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector
        tau : :obj:`float`
            Positive scalar weight

        """
        return self._proxdual_moreau(x, tau, **kwargs)

    def grad(self, x):
        """Compute gradient

        Parameters
        ----------
        x : :obj:`np.ndarray`
            Vector

        Returns
        -------
        g : :obj:`np.ndarray`
            Gradient vector

        """
        pass

    def affine_addition(self, v):
        """Affine addition

        Adds the dot-product of vector ``v`` and vector ``x`` (which is passed
        to ``dual`` or ``proxdual``) to the current function.

        This method can also be accessed via the ``+`` operator.

        Parameters
        ----------
        v : :obj:`np.ndarray`
            Vector

        Notes
        -----
        The proximal operator of a function :math:`g=f(\mathbf{x}) +
        \mathbf{v}^T \mathbf{x}` is defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) =
            prox_{\tau f} (\mathbf{x} - \tau \mathbf{v})

        """
        if isinstance(v, np.ndarray):
            return _SumOperator(self, v)
        else:
            return NotImplemented

    def postcomposition(self, sigma):
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
            return NotImplemented

    def precomposition(self, a, b):
        r"""Precomposition

        Multiplies and add scalars ``a`` and ``b`` to ``x`` when evaluating
        the proximal function

        Parameters
        ----------
        a : :obj:`float`
            Multiplicative scalar
        b : :obj:`float`
            Additive Scalar

        Notes
        -----
        The proximal operator of a function :math:`g= f(a \mathbf{x} + b)` is
        defined as:

        .. math::

            prox_{\tau g} (\mathbf{x}) = \frac{1}{a} (
            prox_{a^2 \tau f} (a \mathbf{x} + b) - b)

        """
        if isinstance(a, float) and isinstance(b, float):
            return _PrecompositionOperator(self, a, b)
        else:
            return NotImplemented

    def chain(self, g):
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

    def __add__(self, v):
        return self.affine_addition(v)

    def __sub__(self, v):
        return self.__add__(-v)

    def __rmul__(self, sigma):
        if isinstance(sigma, (int, float)):
            return self.postcomposition(sigma)
        else:
            return self.chain(sigma)

    #__rmul__ = __mul__

    def _adjoint(self):
        """Adjoint operator - swaps prox and proxdual"""
        return _AdjointOperator(self)

    H = property(_adjoint)


class _AdjointOperator(ProxOperator):
    def __init__(self, f):
        self.f = f
        super().__init__(None, True if f.grad else False)

    def __call__(self, x):
        return self.f(x)

    @_check_tau
    def prox(self, x, tau, **kwargs):
        return self.f.proxdual(x, tau, **kwargs)

    @_check_tau
    def proxdual(self, x, tau, **kwargs):
        return self.f.prox(x, tau, **kwargs)


class _SumOperator(ProxOperator):
    def __init__(self, f, v):
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(v, np.ndarray):
            raise ValueError('Second input must be a numpy array')
        self.f, self.v = f, v
        super().__init__(None, True if f.grad else False)

    def __call__(self, x):
        return self.f(x) + np.dot(self.v, x)

    @_check_tau
    def prox(self, x, tau, **kwargs):
        return self.f.prox(x - tau * self.v, tau)

    def grad(self, x):
        return self.f.grad(x) + self.v


class _ChainOperator(ProxOperator):
    def __init__(self, f, g):
        #if not isinstance(f, ProxOperator) or not isinstance(g, ProxOperator):
        #    raise ValueError('Inputs must be a ProxOperator')
        self.f, self.g = f, g
        super().__init__(None, True if f.grad else False)

    def __call__(self, x):
        pass

    @_check_tau
    def prox(self, x, tau, **kwargs):
        return self.g.prox(self.f.prox(x, tau), tau)

    def grad(self, x):
        pass


class _PostcompositionOperator(ProxOperator):
    def __init__(self, f, sigma):
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(sigma, float):
            raise ValueError('Second input must be a float')
        self.f, self.sigma = f, sigma
        super().__init__(None, True if f.grad else False)

    def __call__(self, x):
        return self.sigma * self.f(x)

    @_check_tau
    def prox(self, x, tau, **kwargs):
        return self.f.prox(x, self.sigma * tau)

    def grad(self, x):
        return self.sigma * self.f.grad(x)


class _PrecompositionOperator(ProxOperator):
    def __init__(self, f, a, b):
        #if not isinstance(f, ProxOperator):
        #    raise ValueError('First input must be a ProxOperator')
        if not isinstance(a, float):
            raise ValueError('Second input must be a float')
        if not isinstance(b, float):
            raise ValueError('Second input must be a float')
        self.f, self.a, self.b = f, a, b
        super().__init__(None, True if f.grad else False)

    def __call__(self, x):
        return self.f(self.a * x + self.b)

    @_check_tau
    def prox(self, x, tau, **kwargs):
        return (self.f.prox(self.a * x + self.b, (self.a ** 2) * tau) -
                self.b) / self.a

    def grad(self, x):
        return self.a * self.f.grad(self.a * x + self.b)