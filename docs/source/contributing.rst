.. _contributing:

|:heart:| Contributing
######################

Contributions are welcome and greatly appreciated!

The best way to get in touch with the core developers and mantainers is to
open new *Issues* directly from the
`github repo <https://github.com/PyLops/pyproximal>`_.

Welcomed contributions
**********************

Bug reports
===========

Report bugs at https://github.com/PyLops/pyproximal/issues.

If you are playing with the PyProximal library and find a bug, please report it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

New operators and features
==========================

Open an issue at https://github.com/PyLops/pyproximal/issues with tag *enhancement*.

If you are proposing a new operator or a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

Fix issues
==========

There is always a backlog of issues that need to be dealt with.
Look through the `GitHub Issues <https://github.com/PyLops/pyproximal/issues>`_ for operator/feature requests or bugfixes.

Add examples or improve documentation
=====================================

Writing new operators is not the only way to get involved and contribute. Create examples with existing operators
as well as improving the documentation of existing operators is as important as making new operators and very much
encouraged.


Step-by-step instructions for contributing
******************************************

Ready to contribute?

1. Follow all instructions in :ref:`DevInstall`.

2. Create a branch for local development, usually starting from the dev branch:

.. code-block:: bash

   >>  git checkout -b name-of-your-branch

Now you can make your changes locally.

3. When you're done making changes, check that your code follows the guidelines for :ref:`addingoperator` and
that both old and new tests pass successfully:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make tests
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make tests_uv
   
   .. tab-item:: :iconify:`material-icon-theme:foxpro` nox

        .. code-block:: bash

            >> make tests_nox
        
        Whilst not enforced, this is recommended as it runs the tests
        with different versions of Python (the same that are used in our CI). 
        Note that you need to have `nox` installed to run this command - 
        use `pipx install nox` or `brew install nox` on macOS to install it.

4. Run ruff to check the quality of your code:

.. tab-set::

   .. tab-item:: :iconify:`devicon:anaconda` conda

        .. code-block:: bash

            >> make lint
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make lint_uv
   
Note that PyProximal enforces full compliance with ruff and it
will also be run as part of our CI.

5. Update the docs

.. tab-set::

   .. tab-item::  conda

        .. code-block:: bash

            >> make docupdate
   
   .. tab-item:: :iconify:`material-icon-theme:uv` uv

        .. code-block:: bash

            >> make docupdate_uv

6. Commit your changes and push your branch to GitLab:

.. code-block:: bash

   >>  git add .
   >> git commit -m "Your detailed description of your changes."
   >> git push origin name-of-your-branch

Remember to add ``-u`` when pushing the branch for the first time.
We recommend using `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/#summary>`_ to
format your commit messages, but this is not enforced.

7. Submit a pull request through the GitHub website.


Pull Request Guidelines
***********************

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests.


Project structure
*****************

This repository is organized as follows:

* **pyproximal**: Python library containing various proximal operators and solvers
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       Sphinx documentation
* **examples**:   set of python script examples for each proximal operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery
