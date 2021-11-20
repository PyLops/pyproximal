import numpy as np

from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class Nonlinear(ProxOperator):
    r"""Nonlinear function proximal operator.

    Proximal operator for a generic nonlinear function :math:`f`. This is a
    template class which a user must subclass and implement the following
    methods:

    - ``fun``: a method evaluating the generic function :math:`f`
    - ``grad``: a method evaluating the gradient of the generic function
      :math:`f`
    - ``optimize``: a method that solves the optimization problem associated
      with the proximal operator of :math:`f`. Note that the
      ``gradprox`` method must be used (instead of ``grad``) as this will
      automatically add the regularization term involved in the evaluation
      of the proximal operator

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
    def __init__(self, x0, niter=10, warm=True):
        super().__init__(None, True)
        self.niter = niter
        self.x0 = x0
        self.warm = warm

    def __call__(self, x):
        return self.fun(x)

    def _funprox(self, x, tau):
        return self.fun(x) + 1. / (2 * tau) * ((x - self.y) ** 2).sum()

    def _gradprox(self, x, tau):
        return self.grad(x) + 1. / tau * (x - self.y)

    def fun(self, x):
        raise NotImplementedError('The method fun has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')
    def grad(self, x):
        raise NotImplementedError('The method grad has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')
    def optimize(self):
        raise NotImplementedError('The method optimize has not been implemented.'
                                  'Refer to the documentation for details on '
                                  'how to subclass this operator.')
    @_check_tau
    def prox(self, x, tau):
        self.y = x
        self.tau = tau
        x = self.optimize()
        if self.warm:
            self.x0 = x
        return x