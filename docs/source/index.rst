PyProximal
==========
This Python library provides all the needed building blocks for solving
non-smooth convex optimization problems using the so-called **proximal algorithms**.

Whereas gradient based methods are first-order iterative optimization
algorithms for solving unconstrained, smooth optimization problems,
proximal algorithms can be viewed as an analogous tool for non-smooth and
possibly constrained versions of these problems. Such algorithms
sit at a higher level of abstraction than classical algorithms
like Steepest descent or Newton’s method and require a basic
operation to be performed at each iteration: the evaluation of the
so-called *proximal operator* of the functional to be optimized.

Whilst evaluating a proximal operator does itself require solving a
convex optimization problem, these subproblems often admit closed form
solutions or can be solved very quickly with ad-hoc specialized methods.
Several of such proximal operators are therefore implemented in this
library.

Here is a simple example showing how to compute the
proximal operator of the L1 norm of a vector:

.. code-block:: python

   import numpy as np
   from pyproximal import L1

   l1 = L1(sigma=1.)
   x = np.arange(-5, 5, 0.1)
   xp = l1.prox(x, 1)

and how this can be used to solve a basic denoising problem of the form:

.. math::

        \arg min_\mathbf{x} \frac{\sigma}{2}
        ||\mathbf{x} - \mathbf{y} ||_2^2 + ||\mathbf{D} \mathbf{x}||_1

.. code-block:: python

   import numpy as np
   from pylops import FirstDerivative
   from pyproximal import L1, L2
   from pyproximal.optimization.primal import LinearizedADMM

   np.random.seed(1)

   # Create noisy data
   nx = 101
   x = np.zeros(nx)
   x[:nx//2] = 10
   x[nx//2:3*nx//4] = -5
   n = np.random.normal(0, 2, nx)
   y = x + n

   # Define functionals
   l2 = L2(b=y)
   l1 = L1(sigma=5.)
   Dop = FirstDerivative(nx, edge=True, kind='backward')

   # Solve functional with L-ADMM
   L = np.real((Dop.H * Dop).eigs(neigs=1, which='LM')[0])
   tau = 1.
   mu = 0.99 * tau / L
   xladmm, _ = LinearizedADMM(l2, l1, Dop, tau=tau, mu=mu,
                              x0=np.zeros_like(x), niter=200)


Why another library for proximal algorithms?
--------------------------------------------

Several other projects in the Python ecosystem provide implementations of proximal
operators and/or algorithms which present some clear overlap with this project.

A (possibly not exahustive) list of other projects is:

* http://proximity-operator.net
* https://github.com/ganguli-lab/proxalgs/blob/master/proxalgs/operators.py
* https://github.com/pmelchior/proxmin
* https://github.com/comp-imaging/ProxImaL
* https://github.com/matthieumeo/pycsou

All of these projects are self-contained, meaning that they implement both proximal
and linear operators as needed to solve a variety of problems in different areas
of science.

The main difference with PyProximal lies in the fact that we decide *not to* intertangle
linear and proximal operators within the same library. We leverage the extensive
set of linear operators provided by the `PyLops <http://pylops.readthedocs.io>`_
project and focus only on the proximal part of the problem. This makes the codebase
more concise, and easier to understand and extend. As explained more in details in
:ref:`addingoperator` section, a new proximal operator can created by simply
subclassing the :py:class:`pyproximal.ProxOperator` class and by implementing
``prox`` and ``proxdual``.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   installation.rst
   tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation:

   api/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved:

   Implementing new operators  <adding.rst>
   Contributing <contributing.rst>
   Changelog <changelog.rst>
   Credits <credits.rst>

