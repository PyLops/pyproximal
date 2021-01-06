.. _installation:

Installation
============

To install the PyProximal library you will need **Python 3.6 or greater**.


Dependencies
------------

Our mandatory dependencies are limited to:

* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org/scipylib/index.html>`_
* `pylops <https://pylops.readthedocs.io>`_

We advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
to ensure that these dependencies are installed via the ``Conda`` package manager. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (i.e., a highly optimized BLAS library created by Intel).


Step-by-step installation for users
-----------------------------------

Python environment
~~~~~~~~~~~~~~~~~~

Activate your Python environment, and simply type the following command in your terminal
to install the PyPi distribution:

.. code-block:: bash

   >> pip install pyproximal

Coming soon!

Alternatively, to access the latest source from github:

.. code-block:: bash

   >> pip install https://github.com/mrava87/pyproximal/archive/master.zip

or just clone the repository

.. code-block:: bash

   >> git clone https://github.com/mrava87/pyproximal.git

or download the zip file from the repository (green button in the top right corner of the
main github repo page) and install PyLops from terminal using the command:

.. code-block:: bash

   >> make install


Step-by-step installation for developers
----------------------------------------
Fork and clone the repository by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/your_name_here/pyproximal.git

The first time you clone the repository run the following command:

.. code-block:: bash

   >> make dev-install

If you prefer to build a new Conda enviroment just for PyLops, run the following command:

.. code-block:: bash

   >> make dev-install_conda

To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

    >> make tests

Make sure no tests fail, this guarantees that the installation has been successfull.

If using Conda environment, always remember to activate the conda environment every time you open
a new *bash* shell by typing:

.. code-block:: bash

   >> source activate pyproximal
