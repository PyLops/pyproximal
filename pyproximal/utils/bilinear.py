from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from abc import ABC, abstractmethod
from pylops.utils.typing import NDArray

if TYPE_CHECKING:
    from pylops.linearoperator import LinearOperator


class BilinearOperator(ABC):
    r"""Common interface for bilinear operator of a function.

    Bilinear operator template class. A user
    must subclass it and implement the following methods:

    - ``gradx``: a method evaluating the gradient over :math:`\mathbf{x}`:
      :math:`\nabla_x H`
    - ``grady``: a method evaluating the gradient over :math:`\mathbf{y}`:
      :math:`\nabla_y H`
    - ``grad``: a method returning the stacked gradient vector over
      :math:`\mathbf{x},\mathbf{y}`: :math:`[\nabla_x H`, [\nabla_y H]`
    - ``lx``: Lipschitz constant of :math:`\nabla_y H`
    - ``ly``: Lipschitz constant of :math:`\nabla_x H`
    - ``updatexy``: Update :math:`\mathbf{x}` and :math:`\mathbf{y}`
      from a single vector :math:`\mathbf{xy}`

    Two additional methods (``updatex`` and ``updatey``) are provided to
    update the :math:`\mathbf{x}` and :math:`\mathbf{y}` internal
    variables. It is user responsability to choose when to invoke such
    methods (i.e., when to update the internal variables).

    Notes
    -----
    A bilinear operator is defined as a differentiable nonlinear function
    :math:`H(x,y)` that is linear in each of its components indipendently,
    i.e, :math:`\mathbf{H_x}(y)\mathbf{x}` and :math:`\mathbf{H_y}(y)\mathbf{x}`.

    """
    def __init__(self) -> None:
        # initialize sizex and sizey, e.g. to 0 to avoid runtime errors 
        # if not set by subclass and accessed
        self.sizex: int = 0
        self.sizey: int = 0
        
        # initialize x and y to empty arrays or placeholder of correct type
        self.x: NDArray = np.array([])
        self.y: NDArray = np.array([])
        pass

    @abstractmethod
    def __call__(self, x: NDArray, y: Optional[NDArray] = None) -> Any:
        pass
    
    @abstractmethod
    def gradx(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def grady(self, y: NDArray) -> NDArray:
        pass

    @abstractmethod
    def grad(self, x_or_y: NDArray) -> NDArray:
        pass

    @abstractmethod
    def lx(self, x: NDArray) -> float:
        pass
    
    @abstractmethod
    def ly(self, y: NDArray) -> float:
        pass
    
    @abstractmethod
    def updatexy(self, xy: NDArray) -> None:
        pass

    def updatex(self, x: NDArray) -> None:
        """Update x variable (to be used to update the internal variable x)
        """
        self.x = x

    def updatey(self, y: NDArray) -> None:
        """Update y variable (to be used to update the internal variable y)
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

        \nabla_x H(\mathbf{x};\ \mathbf{y}) =
        \mathbf{Op}^H(\mathbf{Op}(\mathbf{X}\mathbf{Y})
        - \mathbf{d})\mathbf{Y}^H

    and gradient with respect to y equal to:

    .. math::

        \nabla_y H(\mathbf{y}; \mathbf{x}) =
        \mathbf{X}^H \mathbf{Op}^H(\mathbf{Op}
        (\mathbf{X}\mathbf{Y}) - \mathbf{d})

    Note that in both cases, the currently stored :math`\mathbf{x}`/:math`\mathbf{y}` variable
    is used for the second variable within parenthesis (after ;).

    """
    def __init__(
            self, 
            X: NDArray, 
            Y: NDArray, 
            d: NDArray,
            Op: Optional["LinearOperator"] = None,
            ) -> None:
        self.n, self.k = X.shape
        self.m = Y.shape[1]

        self.x = X
        self.y = Y
        self.d = d
        self.Op = Op
        self.sizex = self.n * self.k
        self.sizey = self.m * self.k

    def __call__(self, x: NDArray, y: Optional[NDArray] = None) -> float:
        # x can be concatenated [x,y] or just x if y is provided
        if y is None:
            x, y = x[:self.n * self.k], x[self.n * self.k:]
        # store original self.x to restore after calculation, 
        # as _matvecy uses self.x
        xold = self.x.copy()
        # temporarily update self.x for _matvecy
        self.updatex(x)
        # compute residual: note that _matvecy(y) computes 
        # Op(X @ Y) where Y is obtained reshaping y into
        # a matrix
        res = self.d - self._matvecy(y)
        # restore original self.x
        self.updatex(xold)
        return float(np.linalg.norm(res)**2 / 2.)

    def _matvecx(self, x: NDArray) -> NDArray:
        # recreate matrix from flattened x
        X = x.reshape(self.n, self.k)
        X = X @ self.y.reshape(self.k, self.m)
        # apply operator to vectorized version of XY
        if self.Op is not None:
            X = self.Op @ X.ravel()
        return X.ravel()
    
    def _matvecy(self, y: NDArray) -> NDArray:
        # recreate matrix from flattened y
        Y = y.reshape(self.k, self.m)
        X = self.x.reshape(self.n, self.k) @ Y
        # apply operator to vectorized version of XY
        if self.Op is not None:
            X = self.Op @ X.ravel()
        return X.ravel()

    def matvec(self, x: NDArray) -> NDArray:
        # check that no ambiguous situation arises due to n==m
        if self.n == self.m:
            raise NotImplementedError('Since n=m, this method'
                                      'cannot distinguish automatically'
                                      'between _matvecx and _matvecy. '
                                      'Explicitely call either of those two methods.')
        if x.size == self.sizex:
            y = self._matvecx(x)
        else:
            y = self._matvecy(x)
        return y

    def lx(self, x: NDArray) -> float:
        if self.Op is not None:
            # Lipschitz constant for grad_y H involves Op.H Op and X.H X.
            # This is non-trivial and depends on Op's norm.
            raise ValueError('lx cannot be computed when using Op')
        # if Op is None, H = 0.5 * ||XY - d||^2. grad_y H = X.H (XY-d)
        # Lipschitz of grad_y H involves ||X.H X||_F or ||X||_2^2
        X = x.reshape(self.n, self.k)
        return float(np.linalg.norm(np.conj(X).T @ X, 'fro'))

    def ly(self, y: NDArray) -> float:
        if self.Op is not None:
            # Lipschitz constant for grad_x H involves Op.H Op and Y Y.H.
            # This is non-trivial and depends on Op's norm.
            raise ValueError('ly cannot be computed when using Op')
        # if Op is None, H = 0.5 * ||XY - d||^2. grad_x H = (XY-d)Y.H
        # Lipschitz of grad_x H involves ||Y Y.H||_F or ||Y||_2^2
        Y = y.reshape(self.k, self.m)
        return float(np.linalg.norm(Y @ np.conj(Y).T, 'fro'))

    def gradx(self, x: NDArray) -> NDArray:
        """Gradient of H wrt x
        
        Computes grad_x H = - Op.H (d - Op(XY)) Y.H
        """
        # compute residual
        r = self.d - self._matvecx(x)
        # apply adjoint of operator if present
        if self.Op is not None:
            r = (self.Op.H @ r).reshape(self.n, self.m)
        else:
            r = r.reshape(self.n, self.m)
        # apply Y.H
        g = -r @ np.conj(self.y.reshape(self.k, self.m).T)
        return g.ravel()

    def grady(self, y: NDArray) -> NDArray:
        """Gradient of H wrt y
        
        Computes grad_y H = - X.H Op.H (d - Op(XY))
        """
        # compute residual
        r = self.d - self._matvecy(y)
        # apply adjoint of operator if present
        if self.Op is not None:
            r = (self.Op.H @ r.ravel()).reshape(self.n, self.m)
        else:
            r = r.reshape(self.n, self.m)
        # apply X.H
        g = -np.conj(self.x.reshape(self.n, self.k).T) @ r
        return g.ravel()

    def grad(self, x: NDArray) -> NDArray:
        gx = self.gradx(x[:self.n * self.k])
        gy = self.grady(x[self.n * self.k:])
        g = np.hstack([gx, gy])
        return g

    def updatexy(self, x: NDArray) -> None:
        self.updatex(x[:self.n * self.k])
        self.updatey(x[self.n * self.k:])
