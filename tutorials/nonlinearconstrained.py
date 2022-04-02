r"""
Nonlinear inversion with box constraints
========================================
In this tutorial we focus on a modification of the `Quadratic program
with box constraints` tutorial where the quadratic function is replaced by a
nonlinear function. For this example we will use the well-known Rosenbrock
function:

    .. math::
        \mathbf{x} = \argmin_\mathbf{x} f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{x}
        \in \mathcal{I}_{\operatorname{Box}}

We will learn how to handle nonlinear functionals in convex optimization, and
more specifically dive into the details of the
:class:`pyproximal.proximal.Nonlinear` operator. This is a template operator
which must be subclassed and used for a specific functional. After doing so, we will
need to implement the following three method: `func` and `grad` and `optimize`.
As the names imply, the first method takes a model vector :math:`x` as input and
evaluates the functional. The second method evaluates the gradient of the
functional with respect to :math:`x`. The third method implements an
optimization routine that solves the proximal operator of :math:`f`,
more specifically:

    .. math::
        \prox_{\tau f} (\mathbf{x}) = \argmin_{\mathbf{y}} f(\mathbf{y}) +
        \frac{1}{2 \tau}\|\mathbf{y} - \mathbf{x}\|^2_2

Note that when creating the ``optimize`` method a user must use the gradient
of the augmented functional which is provided by the `_gradprox` built-in
method in :class:`pyproximal.proximal.Nonlinear` class.

In this example, we will consider both the
:func:`pyproximal.optimization.primal.ProximalGradient` and
:func:`pyproximal.optimization.primal.ADMM` algorithms. The former solver
will simply use the `grad` method whilst the second solver relies on the
`optimize` method.

"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import pyproximal

plt.close('all')

###############################################################################
# Let's start by defining the class for the nonlinear functional

def rosenbrock(x, y, a=1, b=10):
    f = (a - x)**2 + b*(y - x**2)**2
    return f

def rosenbrock_grad(x, y, a=1, b=10):
    dfx = -2*(a - x) - 2*b*(y - x**2) * 2 * x
    dfy = 2*b*(y - x**2)
    return dfx, dfy

def contour_rosenbrock(x, y):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Evaluate the function
    x, y = np.meshgrid(x, y)
    z = rosenbrock(x, y)

    # Plot the surface.
    surf = ax.contour(x, y, z, 200, cmap='gist_heat_r', vmin=-20, vmax=200,
                      antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    return fig, ax

class Rosebrock(pyproximal.proximal.Nonlinear):
    def setup(self, a=1, b=10, alpha=1.):
        self.a, self.b = a, b
        self.alpha = alpha
    def fun(self, x):
        return np.array(rosenbrock(x[0], x[1], a=self.a, b=self.b))
    def grad(self, x):
        return np.array(rosenbrock_grad(x[0], x[1], a=self.a, b=self.b))
    def optimize(self):
        self.solhist = []
        sol = self.x0.copy()
        for iiter in range(self.niter):
            x1, x2 = sol
            dfx1, dfx2 = self._gradprox(sol, self.tau)
            x1 -= self.alpha * dfx1
            x2 -= self.alpha * dfx2
            sol = np.array([x1, x2])
            self.solhist.append(sol)
        self.solhist = np.array(self.solhist)
        return sol


###############################################################################
# We can now setup the problem and solve it without constraints using a simple
# gradient descent with fixed-step size (of course we could choose any other
# solver)

niters = 500
alpha = 0.02

steps = [(0, 0), ]
for iiter in range(niters):
    x, y = steps[-1]
    dfx, dfy = rosenbrock_grad(x, y)
    x -= alpha * dfx
    y -= alpha * dfy
    steps.append((x, y))

x = np.arange(-1.5, 1.5, 0.15)
y = np.arange(-0.5, 1.5, 0.15)
nx, ny = len(x), len(y)

###############################################################################
# Let's now define the box constraint

xbound = np.arange(-1.5, 1.5, 0.01)
ybound = np.arange(-0.5, 1.5, 0.01)
X, Y = np.meshgrid(xbound, ybound, indexing='ij')
xygrid = np.vstack((X.ravel(), Y.ravel()))

lower = 0.6
upper = 1.2
indic = (xygrid > lower) & (xygrid < upper)
indic = indic[0].reshape(xbound.size, ybound.size) & \
        indic[1].reshape(xbound.size, ybound.size)


###############################################################################
# We now solve the constrained optimization using the Proximal gradient solver

fnl = Rosebrock(niter=20, x0=np.zeros(2), warm=True)
fnl.setup(1, 10, alpha=0.02)
ind = pyproximal.proximal.Box(lower, upper)

def callback(x):
    xhist.append(x)

x0 = np.array([0, 0])
xhist = [x0,]
xinv_pg = pyproximal.optimization.primal.ProximalGradient(fnl, ind,
                                                          tau=0.001,
                                                          x0=x0, epsg=1.,
                                                          niter=5000, show=True,
                                                          callback=callback)
xhist_pg = np.array(xhist)

###############################################################################
# And using the ADMM solver

x0 = np.array([0, 0])

xhist = [x0,]
xinv_admm = pyproximal.optimization.primal.ADMM(fnl, ind,
                                                tau=1.,
                                                x0=x0,
                                                niter=30, show=True,
                                                callback=callback)
xhist_admm = np.array(xhist)

###############################################################################
# To conclude it is important to notice that whilst we implemented a vanilla
# gradient descent inside the optimize method, any more advanced solver can
# be used (here for example we will repeat the same exercise using L-BFGS from
# scipy.

class Rosebrock_lbfgs(Rosebrock):
    def optimize(self):
        def callback(x):
            self.solhist.append(x)

        self.solhist = []
        self.solhist.append(self.x0)
        sol = sp.optimize.minimize(lambda x: self._funprox(x, self.tau),
                                   x0=self.x0,
                                   jac=lambda x: self._gradprox(x, self.tau),
                                   method='L-BFGS-B', callback=callback,
                                   options=dict(maxiter=15))
        sol = sol.x

        self.solhist = np.array(self.solhist)
        return sol


fnl = Rosebrock_lbfgs(niter=20, x0=np.zeros(2), warm=True)
fnl.setup(1, 10, alpha=0.02)

x0 = np.array([0, 0])
xhist = [x0,]
xinv_admm_lbfgs = pyproximal.optimization.primal.ADMM(fnl, ind,
                                                      tau=1.,
                                                      x0=x0,
                                                      niter=30, show=True,
                                                      callback=callback)
xhist_admm_lbfgs = np.array(xhist)

###############################################################################
# Finally let's compare the results.

fig, ax = contour_rosenbrock(x, y)
steps = np.array(steps)
ax.plot(steps[:, 0], steps[:, 1], '.-k', lw=2, ms=20, alpha=0.4)
ax.contour(X, Y, indic, colors='k')
ax.scatter(1, 1, c='k', s=300)
ax.plot(xhist_pg[:, 0], xhist_pg[:, 1], '.-b', ms=20, lw=2, label='PG')
ax.plot(xhist_admm[:, 0], xhist_admm[:, 1], '.-g', ms=20, lw=2, label='ADMM')
ax.plot(xhist_admm_lbfgs[:, 0], xhist_admm_lbfgs[:, 1], '.-m', ms=20, lw=2,
        label='ADMM with LBFGS')
ax.set_title('Rosenbrock optimization')
ax.legend()
ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
fig.tight_layout()