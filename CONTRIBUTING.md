# Contributing

Contributions are welcome and greatly appreciated!

The best way to get in touch with the core developers and maintainers is to
join the [PyLops slack channel](https://pylops.slack.com/) as well as
open new *Issues* directly from the GitHub repo.

## Welcomed contributions

### Bug reports

Report bugs at https://github.com/PyLops/pyproximal/issues

If you find a bug, please report it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

### New operators and features

The best way to send feedback is to open an issue at
https://github.com/PyLops/pyproximal/issues
with tag *enhancement*.

If you are proposing a new proximal operator or a new feature:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

### Fix issues
There is always a backlog of issues that need to be dealt with.
Look through the GitHub Issues for operator/feature requests or bugfixes.

### Add examples or improve documentation
Writing new operators and solvers is not the only way to get involved and
contribute. Create examples with existing operators and solvers as well as
improving the documentation of existing operators is as important
as making new operators and very much encouraged.


## Step-by-step instructions for contributing

Ready to contribute?

1. Follow all installation instructions in the [Step-by-step installation for developers](https://pyproximal.readthedocs.io/en/stable/installation.html#step-by-step-installation-for-developers) section of the documentation.

2. Create a branch for local development:
    ```
    git checkout -b name-of-your-branch
    ```
    Now you can make your changes locally.

3. When you're done making changes, check that your code follows the guidelines for [Implementing new operators](https://pyproximal.readthedocs.io/en/stable/adding.html) and that old and new tests pass succesfully:
    ```
    make tests
    ```
    or
    ```
    make tests_uv
    ```

4. Run ruff to check the quality of your code:
    ```
    make lint
    ```
    or
    ```
    make lint_uv
    ```

    Note that PyProximal enforces full compliance with ruff and it will also be run as part of our CI.

5. Update the docs
   ```
   make docupdate
   ```

6. Commit your changes and push your branch to GitHub:
    ```
    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-branch
    ```
    Remember to add ``-u`` when pushing the branch for the first time.
    We recommend using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary)
    to format your commit messages, but this is not enforced.

7. Submit a pull request through the GitHub website.


### Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.


## Project structure
This repository is organized as follows:
* **pyproximal**:     Python library containing various proximal operators and solvers
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       Sphinx documentation
* **examples**:   set of python script examples for each proximal operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery
