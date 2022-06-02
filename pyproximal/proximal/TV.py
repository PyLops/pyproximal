import numpy as np

from copy import deepcopy
from scipy.sparse.linalg import lsqr
from pylops import FirstDerivative, Gradient
from pyproximal.ProxOperator import _check_tau
from pyproximal import ProxOperator


class TV(ProxOperator):
    r"""TV Norm proximal operator.

    Proximal operator for the TV norm defined as: :math:`f(\mathbf{x}) =
    \sigma ||\mathbf{x}||_{\text{TV}}`.

    Parameters
    ----------
    dim : :obj:`int`
        Dimension of the object.
    sigma : :obj:`int`, optional
        Multiplicative coefficient of TV norm
    niter : :obj:`int` or :obj:`func`, optional
        Number of iterations of iterative scheme used to compute the proximal.
        This can be a constant number or a function that is called passing a
        counter which keeps track of how many times the ``prox`` method has
        been invoked before and returns the ``niter`` to be used.
    x0 : :obj:`np.ndarray`, optional
        Initial vector
    warm : :obj:`bool`, optional
        Warm start (``True``) or not (``False``). Uses estimate from previous
        call of ``prox`` method.
    rtol : :obj:`float`, optional
        Relative tolerance for stopping criterion.

    Notes
    -----
    The proximal algorithm is implemented following [1].

    .. [1] Beck, A. and Teboulle, M., "Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems", 2009.
    """
    def __init__(self, dim=2, Op=None, sigma=1.,
        niter=10, x0=None, warm=True, rtol=1e-4, **kwargs):
        super().__init__(Op, True)
        self.dim = dim
        self.Op = Op
        self.sigma = sigma
        self.niter = niter
        self.x0 = x0
        self.warm = warm
        self.count = 0
        self.rtol = rtol
        self.kwargs = kwargs

    def __call__(self, x):
        if self.dim == 1:
            if (x.ndim == 1):
                N = len(x)
            else:
                N = x.shape[0]
            derivOp = FirstDerivative(N, dims=None, dir=0, edge=False,
                        dtype="float64", kind="forward")
            dx = derivOp @ x
            y = np.sum(np.abs(dx), axis=0)

        elif self.dim >= 2:
            y = 0
            grads = []
            gradOp = Gradient(x.shape, edge=False, dtype="float64", kind="forward") 
            grads = gradOp.matvec(x.ravel())
            grads = grads.reshape((self.dim,)+x.shape)
            for g in grads:
                y += np.power(abs(g), 2)
            y = np.sqrt(y)

        return self.sigma * np.sum(y)

    def _increment_count(func):
        """Increment counter
        """
        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)
        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        # define current number of iterations
        if isinstance(self.niter, int):
            niter = self.niter
        else:
            niter = self.niter(self.count)

        gamma = self.sigma * tau
        rtol = self.rtol

        # TODO implement test_gamma
        # Initialization
        sol = x

        if self.dim == 1:
            if (x.ndim == 1):
                N = len(x)
            else:
                N = x.shape[0]
            derivOp = FirstDerivative(N, dims=None, dir=0, edge=False,
                        dtype="float64", kind="forward")
        else: 
            gradOp = Gradient(x.shape, edge=False, dtype="float64", kind="forward") 

        if self.dim == 1:
            r = derivOp @ (x*0) 
            # r = op.grad(x * 0, dim=self.dim, **self.kwargs)
            rr = deepcopy(r)
        elif self.dim == 2:
            r, s = gradOp.matvec( (x*0).ravel()).reshape((self.dim,)+x.shape)
            rr, ss = deepcopy(r), deepcopy(s)
        elif self.dim == 3:
            r, s, k = gradOp.matvec( (x*0).ravel()).reshape((self.dim,)+x.shape)
            rr, ss, kk = deepcopy(r), deepcopy(s), deepcopy(k)
        elif self.dim == 4:
            r, s, k, u = gradOp.matvec( (x*0).ravel()).reshape((self.dim,)+x.shape)
            rr, ss, kk, uu = deepcopy(r), deepcopy(s), deepcopy(k), deepcopy(u)

        if self.dim >= 1:
            pold = r
        if self.dim >= 2:
            qold = s
        if self.dim >= 3:
            kold = k
        if self.dim >= 4:
            uold = u

        told, prev_obj = 1., 0.

        # Initialization for weights
        if self.dim >= 1:
            try:
                wx = self.kwargs["wx"]
            except (KeyError, TypeError):
                wx = 1.
        if self.dim >= 2:
            try:
                wy = self.kwargs["wy"]
            except (KeyError, TypeError):
                wy = 1.
        if self.dim >= 3:
            try:
                wz = self.kwargs["wz"]
            except (KeyError, TypeError):
                wz = 1.
        if self.dim >= 4:
            try:
                wt = self.kwargs["wt"]
            except (KeyError, TypeError):
                wt = 1.

        if self.dim == 1:
            mt = wx
        elif self.dim == 2:
            mt = np.maximum(wx, wy)
        elif self.dim == 3:
            mt = np.maximum(wx, np.maximum(wy, wz))
        elif self.dim == 4:
            mt = np.maximum(np.maximum(wx, wy), np.maximum(wz, wt))


        if self.dim >= 1:
            try:
                rr *= np.conjugate(wx)
            except KeyError:
                pass
        if self.dim >= 2:
            try:
                ss *= np.conjugate(wy)
            except KeyError:
                pass
        if self.dim >= 3:
            try:
                kk *= np.conjugate(wz)
            except KeyError:
                pass
        if self.dim >= 4:
            try:
                uu *= np.conjugate(wt)
            except KeyError:
                pass

        iter = 0
        while iter <= niter:
            # Current Solution
            if self.dim == 0:
                raise ValueError("Need to input at least one value")

            if self.dim >= 1:
                div = np.concatenate((np.expand_dims(rr[0, ], axis=0),
                                    rr[1:-1, ] - rr[:-2, ],
                                    -np.expand_dims(rr[-2, ], axis=0)),
                                axis=0)

            if self.dim >= 2:
                div += np.concatenate((np.expand_dims(ss[:, 0, ], axis=1),
                                    ss[:, 1:-1, ] - ss[:, :-2, ],
                                    -np.expand_dims(ss[:, -2, ], axis=1)),
                                    axis=1)

            if self.dim >= 3:
                div += np.concatenate((np.expand_dims(kk[:, :, 0, ], axis=2),
                                    kk[:, :, 1:-1, ] - kk[:, :, :-2, ],
                                    -np.expand_dims(kk[:, :, -2, ], axis=2)),
                                    axis=2)

            if self.dim >= 4:
                div += np.concatenate((np.expand_dims(uu[:, :, :, 0, ], axis=3),
                                    uu[:, :, :, 1:-1, ] - uu[:, :, :, :-2, ],
                                    -np.expand_dims(uu[:, :, :, -2, ], axis=3)),
                                    axis=3)
            sol = x - gamma * div

            #  Objective function value
            obj = 0.5 * np.power(np.linalg.norm(x[:] - sol[:]), 2) + \
                gamma * np.sum(self.__call__(sol), axis=0)
            if (obj > 1e-10):
                rel_obj = np.abs(obj - prev_obj) / obj
            else:
                rel_obj = 2*rtol
            prev_obj = obj

            # Stopping criterion
            if rel_obj < rtol:
                break

            #  Update divergence vectors and project
            if self.dim == 1:
                dx = derivOp(sol)
                r -= 1. / (4 * gamma * mt**2) * dx
                weights = np.maximum(1, np.abs(r))

            elif self.dim == 2:
                dx, dy = gradOp.matvec( sol.ravel()).reshape((self.dim,)+x.shape)
                r -= (1. / (8. * gamma * mt**2.)) * dx
                s -= (1. / (8. * gamma * mt**2.)) * dy
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2)))

            elif self.dim == 3:
                dx, dy, dz = gradOp.matvec( sol.ravel()).reshape((self.dim,)+x.shape)
                r -= 1. / (12. * gamma * mt**2) * dx
                s -= 1. / (12. * gamma * mt**2) * dy
                k -= 1. / (12. * gamma * mt**2) * dz
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2) +
                                                np.power(np.abs(k), 2)))

            elif self.dim == 4:
                dx, dy, dz, dt = gradOp.matvec( sol.ravel()).reshape((self.dim,)+x.shape)
                r -= 1. / (16 * gamma * mt**2) * dx
                s -= 1. / (16 * gamma * mt**2) * dy
                k -= 1. / (16 * gamma * mt**2) * dz
                u -= 1. / (16 * gamma * mt**2) * dt
                weights = np.maximum(1, np.sqrt(np.power(np.abs(r), 2) +
                                                np.power(np.abs(s), 2) +
                                                np.power(np.abs(k), 2) +
                                                np.power(np.abs(u), 2)))

            # FISTA update
            t = (1 + np.sqrt(4 * told**2)) / 2.

            if self.dim >= 1:
                p = r / weights
                r = p + (told - 1) / t * (p - pold)
                pold = p
                rr = deepcopy(r)

            if self.dim >= 2:
                q = s / weights
                s = q + (told - 1) / t * (q - qold)
                ss = deepcopy(s)
                qold = q

            if self.dim >= 3:
                o = k / weights
                k = o + (told - 1) / t * (o - kold)
                kk = deepcopy(k)
                kold = o

            if self.dim >= 4:
                m = u / weights
                u = m + (told - 1) / t * (m - uold)
                uu = deepcopy(u)
                uold = m

            told = t
            iter += 1

        return sol
