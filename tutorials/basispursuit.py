r"""
Basis Pursuit
=============
This tutorial considers the Basis Pursuit problem. From a mathematical point of view we seek the sparsest solution
that satisfies a system of equations.

.. math::
    arg \;  min_\mathbf{x} ||\mathbf{x}||_1 \; s.t. \; \mathbf{Ax} = \mathbf{y}

where the operator :math:`\mathbf{A}_{(N \times M}` is generally a skinny matrix (:math:`N<M`)
Note that this problem is similar to more general L1-regularized inversion but it presents a stricter condition on the
data term which must be satisfied exactly here.

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

