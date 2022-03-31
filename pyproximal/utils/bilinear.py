import numpy as np


class BilinearOperator():
    r"""Common interface for bilinear operator of a function.

    Bilinear operator template class. A user
    must subclass it and implement the following methods:

    - ``gradx``: a method evaluating the gradient over :math:`\mathbf{x}`:
      :math:`\nabla_x H`
    - ``grady``: a method evaluating the gradient over :math:`\mathbf{y}`:
      :math:`\nabla_y H`
    - ``lx``: Lipschitz constant of :math:`\nabla_x H`
    - ``ly``: Lipschitz constant of :math:`\nabla_y H`

    Notes
    -----
    A bilinear operator is defined as a differentiable nonlinear function
    :math:`H(x,y)` that is linear in each of its components indipendently,
    i.e, :math:`\mathbf{H_x}(y)\mathbf{x}` and :math:`\mathbf{H_y}(y)\mathbf{x}`.

    """
    def __init__(self):
        pass

    def __call__(self, x, y):
        pass

    def gradx(self, x):
        pass

    def grady(self, y):
        pass

    def lx(self, x):
        pass

    def ly(self, y):
        pass

    def updatex(self, x):
        """Update x variable (used when evaluating the gradient over y
        """
        self.x = x

    def updatey(self, y):
        """Update y variable (used when evaluating the gradient over y
        """
        self.y = y


class LowRankFactorizedMatrix(BilinearOperator):
    r"""Low-Rank Factorized Matrix operator.

    Bilinear operator representing the L2 norm of a Low-Rank Factorized
    Matrix defined as: :math:`H(\mathbf{X}, \mathbf{Y}) =
    \frac{1}{2} \|\mathbf{Op}(\mathbf{X}\mathbf{Y}) - \mathbf{d}\|_2^2`,
    where :math:`\mathbf{X}` is a matrix of size  :math:`n \times k`,
    :math:`\mathbf{Y}` is a matrix of size :math:`k \times m`, and
    :math:`\mathbf{Op}` is a linear operator of size :math:`p \times n`.

    Parameters
    ----------
    X : :obj:`numpy.ndarray`
        Left-matrix of size :math:`n \times k`
    Y : :obj:`numpy.ndarray`
        Right-matrix of size :math:`k \times m`
    d : :obj:`numpy.ndarray`
        Data vector
    Op : :obj:`pylops.LinearOperator`, optional
        Linear operator

    Notes
    -----
    The Low-Rank Factorized Matrix operator has gradient with respect to x
    equal to:

    .. math::

        \nabla_x H = \mathbf{Op}^H(\mathbf{Op}(\mathbf{X}\mathbf{Y})
        - \mathbf{d})\mathbf{Y}^H

    and gradient with respect to y equal to:

    .. math::

        \nabla_y H = \mathbf{X}^H \mathbf{Op}^H(\mathbf{Op}
        (\mathbf{X}\mathbf{Y}) - \mathbf{d})

    """
    def __init__(self, X, Y, d, Op=None):
        self.n, self.k = X.shape
        self.m = Y.shape[1]

        self.x = X
        self.y = Y
        self.d = d
        self.Op = Op
        self.shapex = (self.n * self.m, self.n * self.k)
        self.shapey = (self.n * self.m, self.m * self.k)

    def __call__(self, x, y):
        xold = self.x.copy()
        self.updatex(x)
        res = self.d - self._matvecy(y)
        self.updatex(xold)
        return np.linalg.norm(res)**2 / 2.

    def _matvecx(self, x):
        X = x.reshape(self.n, self.k)
        X = X @ self.y.reshape(self.k, self.m)
        if self.Op is not None:
            X = self.Op @ X.ravel()
        return X.ravel()

    def _matvecy(self, y):
        Y = y.reshape(self.k, self.m)
        X = self.x.reshape(self.n, self.k) @ Y
        if self.Op is not None:
            X = self.Op @ X.ravel()
        return X.ravel()

    def matvec(self, x):
        if x.size == self.shapex[1]:
            y = self._matvecx(x)
        else:
            y = self._matvecy(x)
        return y

    def lx(self, x):
        X = x.reshape(self.n, self.k)
        # TODO: not clear how to handle Op
        #if self.Op is not None:
        #    X = self.Op @ X
        return np.linalg.norm(np.conj(X).T @ X, 'fro')

    def ly(self, y):
        Y = np.conj(y.reshape(self.k, self.m)).T
        # TODO: not clear how to handle Op
        #if self.Op is not None:
        #    Y = self.Op.H @ Y
        return np.linalg.norm(np.conj(Y).T @ Y, 'fro')

    def gradx(self, x):
        r = self.d - self.matvec(x)
        if self.Op is not None:
            r = (self.Op.H @ r).reshape(self.n, self.m)
        else:
            r = r.reshape(self.n, self.m)
        g = -r @ self.y.reshape(self.k, self.m).T
        return g.ravel()

    def grady(self, y):
        r = self.d - self.matvec(y)
        if self.Op is not None:
            r = (self.Op.H @ r.ravel()).reshape(self.n, self.m)
        else:
            r = r.reshape(self.n, self.m)
        g = -self.x.reshape(self.n, self.k).T @ r
        return g.ravel()
