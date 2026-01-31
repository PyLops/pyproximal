.. _installation:

|:desktop_computer:| Installation
#################################

Dependencies
************

The mandatory dependencies of PyProximal are limited to:

* Python 3.10 or greater
* `NumPy <http://www.numpy.org>`_
* `SciPy <http://www.scipy.org/scipylib/index.html>`_
* `PyLops <https://pylops.readthedocs.io>`_

We highly encourage using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
or its standalone package manager `Conda <https://docs.conda.io/en/latest/index.html>`_.
Especially for Intel processors, this ensures a higher performance with no configuration (e.g., 
the linking to ``Intel MKL`` library, a highly optimized BLAS library created by Intel).
For learning, however, the standard installation is often good enough.


Step-by-step installation for users
***********************************

From Package Manager
====================
First install `pyproximal` with your package manager of choice.

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            conda install --channel conda-forge pyproximal

        Most of the dependencies (all required and some of the optional) are
        automatically installed for you.
            
   .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install pyproximal
        
        Only the *required* dependencies are installed. To install
        some of the optional dependencies, run:
        
        .. code-block:: bash

            pip install "pyproximal[advanced]"

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            uv add pyproximal
        
        Only the *required* dependencies are installed. To install
        some of the optional dependencies, run:
        
        .. code-block:: bash

            uv add "pyproximal[advanced]"


From Source
===========
To access the latest source from github:

.. tab-set::

   .. tab-item:: :iconify:`devicon:pypi` pip

        .. code-block:: bash

            pip install https://github.com/PyLops/pyproximal.git@dev

   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            uv add git+https://github.com/PyLops/pyproximal.git --branch dev


.. _DevInstall:

Step-by-step installation for developers
****************************************

Fork PyProximal
===============
Fork the `PyProximal repository <https://github.com/PyLops/pyproximal>`_ and clone it by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/YOUR-USERNAME/pyproximal.git

Install dependencies
====================

We recommend installing dependencies into a separate environment.
For that end, we provide a `Makefile` with useful commands for setting up the environment.

Conda (recommended)
-------------------
For a ``conda`` environment, run

.. code-block:: bash

   >> make dev-install_conda # for x86 (Intel or AMD CPUs)
   >> make dev-install_conda_arm # for arm (M-series Mac)

This will create and activate an environment called ``pylops``, with all required and optional dependencies.

Pip
---
If you prefer a ``pip`` installation, we provide the following command

.. code-block:: bash

   >> make dev-install

Note that, differently from the  ``conda`` command, the above **will not** create a virtual environment.
Make sure you create and activate your environment previously.

Run tests
=========
To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

   >> make tests

Make sure no tests fail, this guarantees that the installation has been successful.

Add remote (optional)
=====================
To keep up-to-date on the latest changes while you are developing, you may optionally add
the PyLops repository as a *remote*.
Run the following command to add the PyLops repo as a remote named *upstream*:

.. code-block:: bash

   >> git remote add upstream https://github.com/PyLops/pylops

From then on, you can pull changes (for example, in the dev branch) with:

.. code-block:: bash

   >> git pull upstream dev


Install pre-commit hooks
========================
To ensure consistency in the coding style of our developers we rely on
`pre-commit <https://pre-commit.com>`_ to perform a series of checks when you are
ready to commit and push some changes. This is accomplished by means of git hooks
that have been configured in the ``.pre-commit-config.yaml`` file.

In order to setup such hooks in your local repository, run:

.. code-block:: bash

   >> pre-commit install

Once this is set up, when committing changes, ``pre-commit`` will reject and "fix" your code by running the proper hooks.
At this point, the user must check the changes and then stage them before trying to commit again.

Final steps
===========
PyLops does not enforce the use of a linter as a pre-commit hook, but we do highly encourage using one before submitting a Pull Request.
A properly configured linter (``flake8``) can be run with:

.. code-block:: bash

   >> make lint

In addition, it is highly encouraged to build the docs prior to submitting a Pull Request.
Apart from ensuring that docstrings are properly formatted, they can aid in catching bugs during development.
Build (or update) the docs with:

.. code-block:: bash

   >> make doc

or

.. code-block:: bash

   >> make docupdate
