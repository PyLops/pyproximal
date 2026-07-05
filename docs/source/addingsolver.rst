.. _addingsolver:

|:fire:| Implementing new solvers
#################################

Users are welcome to create new solvers and add them to the PyProximal library.

In this tutorial, we will go through the key steps in the definition of a solver, using a
sligthly simplified version of :py:class:`pyproximal.optimization.cls_primal.ProximalGradient`
as an example.

.. note::
    In case you are already familiar with PyLops and have created a solver before, these guidelines closely
    follow those in the PyLops documentation. We aim to highlight every step where the creation
    of a PyProximal solver differs from that of a PyLops server or steps that are completely new
    for this kind of solvers.

Creating the solver
-------------------
The first thing we need to do is to locate a file containing solvers in the same family of the solver we plan to
include, or create a new file with the name of the solver we would like to implement (or preferably its family).
Note that as the solver will be a class, we need to follow the UpperCaseCamelCase convention for the class itself
but not for the filename.

At this point we can start by importing the modules that will be needed by the solver.
This varies from solver to solver, however you will always need to import the
:py:class:`pyproximal.optimization.basesolver.Solver` which will be used as *parent* class for any of our solvers.
Moreover, we always recommend to import :py:func:`pyproximal.utils.backend.get_array_module` as solvers should be written
in such a way that it can work both with ``numpy`` and ``cupy`` arrays. See later for details.

.. code-block:: python

    import time

    import numpy as np

    from pyproximal.optimization.basesolver import Solver
    from pyproximal.utils.backend import get_array_module


After that we define our new object:

.. code-block:: python

    class ProximalGradient(Solver):

followed by a `numpydoc docstring <https://numpydoc.readthedocs.io/en/latest/format.html/>`__
(starting with ``r"""`` and ending with ``"""``) containing the documentation of the solver. Such docstring should
contain at least a short description of the solver, a ``Parameters`` section with a description of the
input parameters of the associated ``_init__`` method and a ``Notes`` section providing a reference to the original
solver and possibly a concise mathematical explanation of the solver. Take a look at some of the core solver of PyProximal
to get a feeling of the level of details of the mathematical explanation.

As for any Python class, our solver will need an ``__init__`` method. In this case, however, we will just rely on that
of the base class. A single input parameter is passed to the ``__init__`` method and saved as members of our class,
namely the :class:`pyproximal.optimization.callback.Callbacks` object that contains any callback we wish to attach
to the solver. Moreover, two additional parameters are created that contains the counter of the iterations (which
will be incremented every time the ``step`` method is called) and the current time (this is used later to report
the execution time of the solver). Here is the ``__init__`` method of the base class:

.. code-block:: python

    def __init__(self, callbacks=None):
        self.callbacks = callbacks
        self._registercallbacks()
        self.iiter = 0
        self.tstart = time.time()

We can now move onto writing the *setup* of the solver in the method ``setup``. We will need to write
a piece of code that prepares the solver prior to being able to apply a step. As most solvers in PyProximal are designed
as the sum of two terms (usually defined as ``f`` and ``g``), the setup method usually takes two proximal operators 
(generally represented by ``proxf`` and ``proxg``) plus additional parameters as explained below. However, there are some
special cases in which solvers operate on a single proximal, a list of proximals, or a single proximal and some linear
operator plus a data vector; in that case the number and type of the first few inputs of the ``setup`` method may be
different. 

Next the method usually requires the initial guess of the solver ``x0``, alongside various hyperparameters of the solver
--- e.g., step sizes and parameters involved in the stopping criterion. For example in this case we only have nine parameters.
Two of them, ``niter``, which refers to the maximum allowed number of iterations, and ``tol`` to set the 
tolerance on the residual norm (the solver will be stopped if this is smaller than the chosen tolerance) are always present. 
Moreover, we always have the possibility to decide whether we want to operate the solver (in this case its setup part) in verbose
or silent mode. This is driven by the ``show`` parameter. We will soon discuss how to choose what to print on screen in
case of verbose mode (``show=True``).

The setup method can be loosely seen as composed of four parts. First, the inputs are stored as members of the class.
Moreover the type of the ``x0`` vector is checked to evaluate whether to use ``numpy`` or ``cupy`` for algebraic operations
(this is done by ``self.ncp = get_array_module(x0)``). As some of the optional parameters hyperparameters may require 
some specialized checking and initialization, this is carried out in this second part (see specific solver implementations
for more details). Next, the starting guess or guesses are initialized (see also note below). Finally, a number
of variables are initialized to be used inside the ``step`` method to keep track of the optimization process. Moreover,
note that the ``setup`` method returns the created starting guess ``x`` or guesses (they are not stored as member of the
class).

.. code-block:: python

    def setup(self, proxf, proxg, x0, epsg, tau,
              niter=10, tol=None):
        self.proxf = proxf
        self.proxg = proxg
        self.x0 = x0
        self.epsg = epsg
        self.tau = tau
        self.niter = niter
        self.tol = tol

        self.ncp = get_array_module(x0)

        # more specialized initializations if needed
        # ...
        #

        # initialize solver
        x = x0.copy()
        y = x.copy()

        # create variables to track the objective function and iterations
        self.pfg, self.pfgold = np.inf, np.inf
        self.cost: list[float] = []
        self.tolbreak = False
        self.iiter = 0

        # print setup
        if show:
            self._print_setup(epsg_print, np.iscomplexobj(x0))
        return x, y

.. note::
    For most solvers with multiple proximal operators, more than one initial guess is usually required.
    Alongside ``x0``, such solvers may have an additional (optional) input parameter called ``z0``. In
    this case, we provide a method called ``_x0z0_init`` which takes ``x0`` and ``z0`` and provides
    checks as well as default initialization of ``z0`` when not provided (i.e., ``z0=None``).

At this point, we need to implement the core of the solver, the ``step`` method. Here, we take the input (or inputs for solvers
with multiple variables) at the previous iterate, update it following the rule of the solver of choice, and return it (or them).
The other input parameter required by this method is ``show`` to choose whether we want to print a report of the step on screen or
not. However, if appropriate, a user can add additional input parameters. A simplified version of the step method for
ProximalGradient is:

.. code-block:: python

    def step(self, x, y, show=False):
        xold = x.copy()

        # proximal step
        x = self.proxg.prox(
            x - self.tau * self.proxf.grad(x),
            self.epsg * self.tau
        )
        
        # tolerance check
        if self.tol is not None:
            self.pfgold = self.pfg
            pf, pg = self.proxf(x), self.proxg(x)
            self.pfg = pf + self.epsg * pg
            if np.abs(1.0 - self.pfg / self.pfgold) < self.tol:
                self.tolbreak = True
        else:
            pf, pg = 0.0, 0.0

        self.iiter += 1
        if show:
            self._print_step(x, pf, pg)
        if self.tol is not None or show:
            self.cost.append(float(self.pfg))
        return x, y

Similarly, we also implement a ``run`` method that is in charge of running a number of iterations by repeatedly
calling the ``step`` method. 

.. code-block:: python

    def run(self, x, y, niter, show, itershow):
        while self.iiter < niter and not self.tolbreak:
            x, y = self.step(x, y, showstep)
            self.callback(x)
            # check if any callback has raised a stop flag
            stop = _callback_stop(self.callbacks)
            if stop:
                break
        return x, y

It is worth noting that any number of callbacks can be attached to the solver; some of these
callbacks can implement a stopping criterion and set the ``stop`` member to True when a given
condition is met. The ``_callback_stop`` method is in change of checking if any of the callbacks
has set ``stop`` to True and in the case break the iterations.

Finally, it is also usually convenient to implement a ``finalize`` method; this method can do any required post-processing that should
not be applied at the end of each step, rather at the end of the entire optimization process. For ProximalGradient this is not implemented
as the method of the baseclass already performs the needed steps.

Last but not least, we can wrap it all up in the ``solve`` method. This method takes as input the data, the initial
model and the same hyperparameters of the setup method and runs the entire optimization process. For CG:

.. code-block:: python

    def solve(self, proxf, proxg, x0, epsg, tau,
              niter=10, tol=None, show=False,
              itershow=(10, 10, 10)):
        x, y = self.setup(
            proxf=proxf,
            proxg=proxg,
            x0=x0,
            epsg=epsg,
            tau=tau,
            niter=niter,
            tol=tol,
            show=show,
        )

        x, y = self.run(x, y, niter, show=show, itershow=itershow)
        self.finalize(81, show)
        return x, y, self.iiter, self.cost

And that's it, we have implemented our first solver operator!

Although the methods that we just described are enough to implement any solver of choice, we find important to provide
users with feedback during the inversion process. Imagine that the modelling operator is very expensive and can take
minutes (or even hours to run), we don't want to leave a user waiting for hours before they can tell if the solver has
done something meaningful. To avoid such scenario, we can implement so called `_print_*` methods where
``*=solver, setup, step, finalize`` that print on screen some useful information (e.g., first value of the current
estimate, norm of residual, etc.). The ``solver`` and ``finalize`` print are already implemented in the base class,
the other two must be implemented when creating a new solver. When these methods are implemented and a user passes
``show=True`` to the associated method, our solver will provide such information on screen throughout the inverse
process. To better understand how to write such methods, we suggest to look into the source code of the CG method.

Finally, to be backward compatible with versions of PyProximal, we also want to create a function with the same
name of the class-based solver (but in small letters) which simply instantiates the solver and runs it. This function
is usually placed in the same file of the class-based solver and snake_case should be used for its name.
This function generally takes all the mandatory and optional parameters of the solver as
input and returns some of the most valuable properties of the class-based solver object. An example
for `ProximalGradient` is:

.. code-block:: python

    def ProximalGradient(proxf, proxg, x0, epsg, tau, niter=10,
                         tol=None, callback=None, show=False):
        pgsolve = cProximalGradient(callbacks=[CostToInitialCallback(rtol), ])
        if callback is not None:
            cgsolve.callback = callback
        x, _, _, _ = cgsolve.solve(
            proxf=proxf, proxg=proxg, x0=x0, epsg=epsg, tau=tau,
            niter=niter, tol=tol, show=show,
        )
        return x


Testing the solver
------------------
Being able to write a solver is not yet a guarantee of the fact that the solver is correct, or in other words
that the solver can converge to a correct solution.

We encourage to create a new test within an existing ``test_*.py`` file in the ``pytests`` folder (or in a new file).
We also encourage to test the function-bases solver, as this will implicitly test the underlying class-based solver.

Generally a test file will start with a number of dictionaries containing different parameters we would like to
use in the testing of one or more solvers. The test itself starts with a *decorator* that contains a list
of all (or some) of dictionaries that will would like to use for our specific operator, followed by
the definition of the test:

.. code-block:: python

    @pytest.mark.parametrize("par", [(par1),(par2)])
    def test_ProximalGradient(par):

However, for proximal solvers we often find it difficult to create tests that verify a single solver in isolation
against an expected (or ground-truth) solution due to the convex nature of the objective function. So a strategy
that is commonly used in our test suite is to test two solvers that are capable of solving the same objective
function against each other and verify that they converge to the same solution within some expected tolerance.
We encourage to check our existing tests for inspiration.


Documenting the solver
----------------------
Once the solver has been created, we can add it to the documentation of PyProximal. To do so, simply add the name of
the operator within the ``index.rst`` file in ``docs/source/api`` directory.

Moreover, in order to facilitate the user of your operator by other users, each test should be included in at least
one of the tutorials as part of the Sphinx-gallery of the documentation of the PyProximal library. The directory
``tutorials`` contains currently available tutorials; if the new solver does not fit any of them, create a new tutorial.


Final checklist
---------------
Before submitting your new solver for review, use the following **checklist** to ensure that your code
adheres to the guidelines of PyProximal:

- you have added the new solver to a new or existing file in the ``optimization`` directory within the ``pyproximal``
  package.

- the new class contains at least ``__init__``, ``setup``, ``step``, ``run``, ``finalize``, and ``solve`` methods.

- each of the above methods have a `numpydoc docstring <https://numpydoc.readthedocs.io/>`__ documenting
  at least the input ``Parameters`` and the ``__init__`` method contains also a ``Notes`` section providing a
  mathematical explanation of the solver.

- a new test has been added to an existing ``test_*.py`` file within the ``pytests`` folder. The test should verify
  that the new solver converges to the true solution for a well-designed inverse problem (i.e., full rank operator).

- the new solver is used within at least one *tutorial* (in ``tutorials`` directory).

