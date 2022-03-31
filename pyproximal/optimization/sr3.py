import numpy as np
import pylops
from scipy.sparse.linalg import lsqr as sp_lsqr


def _lsqr(Op, data, iter_lim, z_old, x0, kappa, eps, Reg):
    r"""LSQR

    This function uses LSQR to solve the inner iteration of SR3, given by

    .. math::
        \min_x \dfrac{1}{2}\Vert\begin{bmatrix}A\\ \sqrt{kappa}L\end{bmatrix}x
        - \begin{bmatrix}b\\ \sqrt{kappa}w\Vert_2^2

    LSQR is stopped when the maximum amount of iterations are reached or when
    the stopping criterion is satisfied. There are no other stopping criteria,
    hence the solver should not be used outside of the context of solving the
    inner iteration of SR3.

    Parameters
    ----------
    Op: :obj:`pylops.LinearOperator`
        Forward operator. This is the stacked operator
        :math:`\begin{bmatrix}A\\ \sqrt{kappa}L\end{bmatrix}`
    data: :obj:`numpy.ndarray`
        Data
    iter_lim: :obj:`int`
        Maximum number of iterations
    z_old: :obj:`np.ndarray`
        The previous outer iteration
    x0: :obj:`numpy.ndarray`
       initial guess
    kappa: :obj:`float`
        The regularization parameter for the inner iteration
    eps: :obj:`float`
        The regularization parameter for the outer iteration
    Reg: :obj:`np.ndarray`
        The regularization operator L

    Returns
    -------
        x: :obj:`numpy.ndarray`
            Approximate solution

    """
    data -= Op.matvec(x0)
    x = x0
    beta = np.linalg.norm(data)
    u = data / beta
    v = Op.rmatvec(u)
    alpha = np.linalg.norm(v)
    v = v / alpha
    w = v
    phi_bar = beta
    rho_bar = alpha
    for _ in range(iter_lim):
        u = Op.matvec(v) - alpha*u
        beta = np.linalg.norm(u)
        u = u / beta
        v = Op.rmatvec(u) - beta*v
        alpha = np.linalg.norm(v)
        v = v / alpha
        rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar/rho
        s = beta/rho
        theta = s*alpha
        rho_bar = -c*alpha
        phi = c*phi_bar
        phi_bar = s*phi_bar
        x += (phi/rho)*w
        temp = Reg.matvec(x)
        z = np.sign(temp) * np.maximum(abs(temp) - eps/kappa, 0)
        if np.linalg.norm(z - z_old) < 1e-6 * np.linalg.norm(z_old):
            return x
        w = v - (theta/rho)*w
        z_old = z
    return x


def SR3(Op, Reg, data, kappa, eps, x0=None, adaptive=True,
        iter_lim_outer=int(1e2), iter_lim_inner=int(1e2)):
    r"""Sparse Relaxed Regularized Regression

    Applies the Sparse Relaxed Regularized Regression (SR3) algorithm to
    an inverse problem with a sparsity constraint of the form

    .. math::

        \min_x \dfrac{1}{2}\Vert \mathbf{Ax} - \mathbf{b}\Vert_2^2 +
        \lambda\Vert \mathbf{Lx}\Vert_1

    SR3 introduces an auxiliary variable :math:`\mathbf{z} = \mathbf{Lx}`,
    and instead solves

    .. math::

        \min_{\mathbf{x},\mathbf{z}} \dfrac{1}{2}\Vert \mathbf{Ax} -
        \mathbf{b}\Vert_2^2 + \lambda\Vert \mathbf{z}\Vert_1 +
        \dfrac{\kappa}{2}\Vert \mathbf{Lx} - \mathbf{z}\Vert_2^2

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Forward operator
    Reg : :obj:`numpy.ndarray`
        Regularization operator
    data : :obj:`numpy.ndarray`
        Data
    kappa : :obj:`float`
        Parameter controlling the difference between :math:`\mathbf{z}`
        and :math:`\mathbf{Lx}`
    eps : :obj:`float`
        Regularization parameter
    x0 : :obj:`numpy.ndarray`, optional
        Initial guess
    adaptive : :obj:`bool`, optional
        Use adaptive SR3 with a stopping criterion for the inner iterations
        or not
    iter_lim_outer : :obj:`int`, optional
        Maximum number of iterations for the outer iteration
    iter_lim_inner : :obj:`int`, optional
        Maximum number of iterations for the inner iteration

    Returns
    -------
    x: :obj:`numpy.ndarray`
        Approximate solution.

    Notes
    -----
    SR3 uses the following algorithm:

        .. math::
            \mathbf{x}_{k+1} = (\mathbf{A}^T\mathbf{A} + \kappa
            \mathbf{L}^T\mathbf{L})^{-1}(\mathbf{A}^T\mathbf{b} +
            \kappa \mathbf{L}^T\mathbf{y}_k) \\
            \mathbf{y}_{k+1} = \prox_{\lambda/\kappa\mathcal{R}}
            (\mathbf{Lx}_{k+1})

    """
    (m, n) = Op.shape
    if x0 is None:
        x0 = np.zeros(n)
    x = x0.copy()
    p = Reg.shape[0]
    v = np.zeros(p)
    w = v
    eta = 1/kappa
    theta = 1
    AL = pylops.VStack([Op, np.sqrt(kappa) * Reg])
    for _ in range(iter_lim_outer):
        # Compute the inner iteration
        if adaptive:
            x = _lsqr(AL, np.concatenate((data, np.sqrt(kappa)*v)),
                      iter_lim_inner, v, x, kappa, eps, Reg)
        else:
            x = sp_lsqr(AL, np.concatenate((data, np.sqrt(kappa)*v)),
                        iter_lim=iter_lim_inner, x0=x)[0]
        # Compute the outer iteration
        w_old = w
        temp = Reg.matvec(x)
        w = np.sign(temp) * np.maximum(abs(temp) - eta*eps, 0)
        err1 = np.linalg.norm(v - w) / max(1, np.linalg.norm(w))
        if err1 < 1e-6:
            return x
        theta = 2/(1 + np.sqrt(1+4/(theta**2)))
        v = w + (1-theta)*(w - w_old)
    return x
