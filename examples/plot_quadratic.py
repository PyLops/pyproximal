r"""
Quadratic
=========
In this example we consider the proximal operator for a quadratic function:

.. math::

   \frac{1}{2} \mathbf{x}^T \mathbf{Op} \mathbf{x} + \mathbf{b}^T
   \mathbf{x} + c.

which is implemented by the :class:`pyproximal.Quadratic` class.

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops

import pyproximal

plt.close('all')

###############################################################################
# To start with cosider the most complete case when both :math:`\mathbf{Op}` and
# :math:`\mathbf{Op}` are non-null.
x = np.arange(-5, 5, 0.1)
nx = len(x)

A = np.random.normal(0, 1, (nx, nx))
A = A.T @ A
c = 2.
quad = pyproximal.Quadratic(Op=pylops.MatrixMult(A), b=np.ones_like(x), c=c,
                            niter=500)
print('1/2 x^T Op x + b^T x + c: ', quad(x))

tau = 4
xp = quad.prox(x, tau)
xdp = quad.proxdual(x, tau)

plt.figure(figsize=(7, 2))
plt.plot(x, x, 'k', lw=2, label='x')
plt.plot(x, xp, 'r', lw=2, label='prox(x)')
plt.plot(x, xdp, 'b', lw=2, label='dualprox(x)')
plt.xlabel('x')
plt.title(r'$\frac{1}{2} \mathbf{x}^T \mathbf{Op} \mathbf{x} + '
          r'\mathbf{b}^T \mathbf{x} + c$')
plt.legend()
plt.tight_layout()


###############################################################################
# If we now assume that the operator :math:`\mathbf{Op}` is null, the quadratic
# operator can be used to define the dot-product between :math:`\mathbf{x}` and
# a vector :math:`\mathbf{b}`
x = np.arange(-5, 5, 0.1)

dot = pyproximal.Quadratic(b=np.ones_like(x))
print('b^T x: ', quad(x))

tau = 2
xp = dot.prox(x, tau)
xdp = dot.proxdual(x, tau)

plt.figure(figsize=(7, 2))
plt.plot(x, x, 'k', lw=2, label='x')
plt.plot(x, xp, 'r', lw=2, label='prox(x)')
plt.plot(x, xdp, 'b', lw=2, label='dualprox(x)')
plt.xlabel('x')
plt.title(r'$\mathbf{b}^T \mathbf{x}$')
plt.legend()
plt.tight_layout()

###############################################################################
# Finally if also :math:`\mathbf{b}` is zero, the quadratic function reduces
# to a constant :math:`\mathbf{c}` and its proximity operator becomes the vector
# :math:`\mathbf{x}` itself.
x = np.arange(-5, 5, 0.1)

dot = pyproximal.Quadratic(c=5.)
print('c: ', quad(x))

tau = 2
xp = dot.prox(x, tau)
xdp = dot.proxdual(x, tau)

plt.figure(figsize=(7, 2))
plt.plot(x, x, 'k', lw=2, label='x')
plt.plot(x, xp, 'r', lw=2, label='prox(x)')
plt.plot(x, xdp, 'b', lw=2, label='dualprox(x)')
plt.xlabel('x')
plt.title(r'$c$')
plt.legend()
plt.tight_layout()