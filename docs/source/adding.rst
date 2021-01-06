.. _addingoperator:

Implementing new operators
==========================
Users are welcome to create new operators and add them to the PyProximal library.

In this tutorial, we will go through the key steps in the definition of an operator, using the
:py:class:`pyproximal.L1` as an example. This is a very simple operator that represents the L1 norm
and can be used to compute the proximal and its dual for the L1 norm of a vector ``x``.


Creating the operator
---------------------
The first thing we need to do is to create a new file with the name of the operator we would like to implement.
Note that as the operator will be a class, we need to follow the UpperCaseCamelCase convention both for the class itself
and for the filename.

Once we have created the file, we will start by importing the modules that will be needed by the operator.
While this varies from operator to operator, you will always need to import the :py:class:`pyproximal.ProxOperator` class,
which will be used as *parent* class for any of our operators:

.. code-block:: python

   from pyproximal import ProxOperator

After that we define our new object:

.. code-block:: python

   class L1(ProxOperator):

followed by a `numpydoc docstring <https://numpydoc.readthedocs.io/en/latest/format.html/>`_
(starting with ``r"""`` and ending with ``"""``) containing the documentation of the operator. Such docstring should
contain at least a short description of the operator, a ``Parameters`` section with a detailed description of the
input parameters and a ``Notes`` section providing a mathematical explanation of the operator. Take a look at
some of the core operators of PyProximal to get a feeling of the level of details of the mathematical explanation.

We then need to create the ``__init__`` method where the input parameters are passed and saved as members of our class.
Input parameters change from operator to operator, however two of them are common to most operators and can be passed
directly to the ``super`` method that invokes the ``__init__`` of the parent class. The first one,
called ``Op`` can be used to equip the proximal operator with a PyLops linear operator in case
one is required. Moreover, if we can compute the gradient of the functional associated to the proximal
operator that we want our class to represent, we can pass the ``hasgrad`` flag and choose it to be ``True``.
If this is the case, we then need to implement the ``grad`` method where the gradient is computed and returned.

In our example, as no linear operator is required in our implementation of
the proximal operator of a L1 norm we will pass ``None``. We also pass ``False`` to the ``hasgrad`` flag as we
know that the L1 norm is non-differentiable (around zero). Two additional inputs, namely ``sigma`` and ``g``,
can also be provided by the user. The first one represents a scaler applied to the norm, whilst the second is
a vector to be subtracted to ``x`` inside the norm. Note that apart from storing ``sigma`` and ``g``
inside member variables we also define two additional variables ``gdual`` and ``box`` which will be needed
to implement
the dual of the proximal operator.

.. code-block:: python

    def __init__(self, sigma=1., g=None):
        super().__init__(None, False)
        self.sigma = sigma
        self.g = g
        self.gdual = 0 if g is None else g
        self.box = BoxProj(-sigma, sigma)

The first method that we will be required to implement is the ``__class__`` method. This method can be called to
evaluate the functional that our operator implements given an input vector ``x``, in the case the L1 norm.

.. code-block:: python

    def __call__(self, x):
        return self.sigma * np.sum(np.abs(x))

We can then move onto writing the *proximal operator* in the method ``prox``.
Such method is always composed of the inputs (the object itself ``self``, the input vector  ``x``,
and the scalar coefficience model ``tau``). In this case the code to be added to the forward is very simple,
as the proximal of the L1 norm is a soft-thresholding to be applied to each element of ``x``. Such a tresholding
could be implemented directly here, but as it may be useful in other cases it is implemented by an external method
that we call. We finally need to ``return`` the result of this operation:

.. code-block:: python

    @_check_tau
    def prox(self, x, tau):
        x = _softthreshold(x, tau * self.sigma, self.g)
        return x

Note the ``@_check_tau`` decorator. Such decorator should be added to every proximal and dual proximal methods. This
ensures that if ``tau`` is zero or negative an error will be raised before any computation is performed.

Finally we can also implement the dual of the proximal operator. Such a method is very useful and required by the
so-called primal-dual solvers. However, it is not always easy to find an analytical expression for the dual of the
proximal operator of a functional. If that is the case, we can simply omit this method. If the user calls it, an
indirect implementation of it will be triggered (by the definition of ``proxdual``) in the base class which is based
on the so-called Moreau identity. In this case we have a closed form, which corresponds the orthogonal projection
of a box from ``-sigma`` to ``sigma`` as defined in the ``__init__`` method. Our ``proxdual`` is therefore written as:

.. code-block:: python

    @_check_tau
    def proxdual(self, x, tau):
        x = self.box(x - self.gdual)
        return x


Testing the operator
--------------------
Being able to write an operator is not yet a guarantee of the fact that the operator is correct. Testing proximal
operators is however not easy. Two different scenarios can be idetified:

- a closed form is available for both the proximal and the dual proximal. In this case, we can directly implement
  both of them and use the Moreau identity (:func:`pyproximal.utils.moreau`) to validate their correctness:

    .. math::
        \mathbf{x} = prox_{\tau f} (\mathbf{x}) +
            \tau prox_{\frac{1}{\tau} f^*} (\frac{\mathbf{x}}{\tau})

- a closed form is not available for either the proximal or the dual proximal. In this case, we cannot validate one
  implementation against the other and we need to rely on ad-hoc tests to validate the implementation that we have
  of either of the method. Here it is suggested to consider some edges cases where we know the expected result of
  applying the proximal or dual proximal to a vector and validate that we get the numbers that we expect.

Either way, all you need to do is create a new test within an existing ``test_*.py`` file in the
``pytests`` folder (or in a new file).

Generally a test file will start with a number of dictionaries containing different parameters we would like to
use in the testing of one or more operators. The test itself starts with a *decorator* that contains a list
of all (or some) of dictionaries that will would like to use for our specific operator, followed by
the definition of the test

.. code-block:: python

    @pytest.mark.parametrize("par", [(par1),(par2)])
    def test_L1(par):


At this point we can first of all create the operator, compute the norm, and validate the proximal and dual proximal implementations
via the :py:func:`pyproximal.utils.moreau` preceded by the `assert`` command.

.. code-block:: python

    """L1 norm and proximal/dual proximal
    """
    l1 = L1(sigma=par['sigma'])

    # norm
    x = np.random.normal(0., 1., par['nx']).astype(par['dtype'])
    assert l1(x) == par['sigma'] * np.sum(np.abs(x))

    # prox / dualprox
    tau = 1
    assert moreau(l1, x, tau)


Documenting the operator
------------------------
Once the operator has been created, we can add it to the documentation of PyProximal. To do so, simply add the name of
the operator within the ``index.rst`` file in ``docs/source/api`` directory.

Moreover, in order to facilitate the user of your operator by other users, a simple example should be provided as part of the
Sphinx-gallery of the documentation of the PyProximal library. The directory ``examples`` containes several scripts that
can be used as template.


Final checklist
---------------
Before submitting your new operator for review, use the following **checklist** to ensure that your code
adheres to the guidelines of PyLops:

- you have created a new file containing one or multiple classes and added to a new or existing
  directory within the ``pyproximal`` package.

- the new class contains at least ``__init__``,  ``__call__``, and ``prox``, and optionally
  ``proxdual`` and ``grad`` methods.

- the new class (or function) has a `numpydoc docstring <https://numpydoc.readthedocs.io/>`_ documenting
  at least the input ``Parameters`` and with a ``Notes`` section providing a mathematical explanation of the operator

- a new test has been added to an existing ``test_*.py`` file within the ``pytests`` folder.

- the new operator is used within at least one *example* (in ``examples`` directory) or one *tutorial*
  (in ``tutorials`` directory).