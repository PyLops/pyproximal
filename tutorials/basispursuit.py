r"""
Basis Pursuit
=============
This tutorial considers the Basis Pursuit problem. From a mathematical point of view we seek the sparsest solution
that satisfies a system of equations.

.. math::
    \argmin_\mathbf{x} \|\mathbf{x}\|_1 \; \text{s.t.} \; \mathbf{Ax} = \mathbf{y}

where the operator :math:`\mathbf{A}` is of size :math:`N \times M`, and generally :math:`N<M`.
Note that this problem is similar to more general L1-regularized inversion, but it presents a stricter condition
on the data term which must be satisfied exactly. Similarly, we can also consider the Basis Pursuit Denoise problem

.. math::
    \argmin_\mathbf{x} \|\mathbf{x}\|_1 \; \text{s.t.} \;  \|\mathbf{Ax} - \mathbf{y}\|_2 < \epsilon

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops
from scipy import misc

import pyproximal

plt.close('all')
np.random.seed(10)

###############################################################################
# Let's start by creating the input vector x and operator A, and data y
n, m = 40, 100
mava = 10

# model
x = np.zeros(m)
iava = np.random.permutation(np.arange(m))[:mava]
x[iava] = np.random.normal(0, 1, mava)

# operator
A = np.random.normal(0, 1, (n, m))
Aop = pylops.MatrixMult(A)
y = Aop * x

###############################################################################
# This problem can be solved using the ADMM solver where f is an Affine set and
# g is the L1 norm
f = pyproximal.AffineSet(Aop, y, niter=20)
g = pyproximal.L1()

xinv_early = pyproximal.optimization.primal.ADMM(f, g, np.zeros_like(x),
                                                 0.1, niter=10, show=True)[0]

xinv = pyproximal.optimization.primal.ADMM(f, g, np.zeros_like(x),
                                           0.1, niter=150, show=True)[0]

fig, axs = plt.subplots(1, 2, figsize=(12, 3))
axs[0].plot(x, 'k')
axs[0].plot(xinv_early, '--r')
axs[0].plot(xinv, '--b')
axs[0].set_title('Model')
axs[1].plot(y, 'k', label='True')
axs[1].plot(Aop * xinv_early, '--r', label='Early Inv')
axs[1].plot(Aop * xinv, '--b', label='Inv')
axs[1].set_title('Data')
axs[1].legend()
plt.tight_layout()

###############################################################################
# We can observe how even after few iterations, despite the solution is not
# yet converged the data reconstruction is perfect. This is consequence of the
# fact that for the Basis Pursuit problem the data term is a hard constraint.
# Finally let's consider the case with a soft constraint.

f = pyproximal.L1()
g = pyproximal.proximal.EuclideanBall(y, .1)

L = np.real((Aop.H @ Aop).eigs(1))[0]
tau = .99
mu = tau / L

xinv_early = pyproximal.optimization.primaldual.PrimalDual(f, g, Aop, np.zeros_like(x),
                                                           tau, mu, niter=10)

xinv = pyproximal.optimization.primaldual.PrimalDual(f, g, Aop, np.zeros_like(x),
                                                     tau, mu, niter=1000)

fig, axs = plt.subplots(1, 2, figsize=(12, 3))
axs[0].plot(x, 'k')
axs[0].plot(xinv_early, '--r')
axs[0].plot(xinv, '--b')
axs[0].set_title('Model')
axs[1].plot(y, 'k', label='True')
axs[1].plot(Aop * xinv_early, '--r', label='Early Inv')
axs[1].plot(Aop * xinv, '--b', label='Inv (Res=%.2f)' % np.linalg.norm(y - Aop @ xinv))
axs[1].set_title('Data')
axs[1].legend()
plt.tight_layout()

###############################################################################
# Note that at convergence the norm of the residual of the solution adheres
# to the EuclideanBall constraint