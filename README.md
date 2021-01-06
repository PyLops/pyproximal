# PyProximal

[![Build Status](https://travis-ci.com/PyLops/pylops-gpu.svg?branch=master)](https://travis-ci.com/PyLops/pyproximal)
[![AzureDevOps Status](https://dev.azure.com/matteoravasi/PyLops/_apis/build/status/PyLops.pyproximal?branchName=main)](https://dev.azure.com/matteoravasi/PyLops/_build/latest?definitionId=10&branchName=main)
![GithubAction Status](https://github.com/PyLops/pyproximal/workflows/PyProx/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/pyproximal/badge/?version=latest)](https://pyproximal.readthedocs.io/en/latest/?badge=latest)
[![OS-support](https://img.shields.io/badge/OS-linux,osx-850A8B.svg)](https://github.com/PyLops/pyproximal)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)


:vertical_traffic_light: :vertical_traffic_light: This library is under early development.
Expect things to constantly change until version v1.0.0. :vertical_traffic_light: :vertical_traffic_light:


## Objective
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
```python
import numpy as np
from pyproximal import L1

l1 = L1(sigma=1.)
x = np.arange(-5, 5, 0.1)
xp = l1.prox(x, 1)
```
and how this can be used to solve a basic denoising problem of the form:
``min ||x - y||_2^2 + ||Dx||_1``:

```python
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
```

## Why another library for proximal algorithms?

Several other projects in the Python ecosystem provide implementations of proximal
operators and/or algorithms, which present some clear overlap with this project.

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
set of linear operators provided by the [PyLops](http://pylops.readthedocs.io)
project and focus only on the proximal part of the problem. This makes the codebase
more concise, and easier to understand and extend. Moreover many of the
problems that are solved in PyLops can now be also solved by means of
proximal algorithms!


## Project structure
This repository is organized as follows:
* **pyproximal**: python library containing various orthogonal projections, proximial operators, and solvers
* **pytests**:    set of pytests
* **testdata**:   sample datasets used in pytests and documentation
* **docs**:       sphinx documentation
* **examples**:   set of python script examples for each proximal operator to be embedded in documentation using sphinx-gallery
* **tutorials**:  set of python script tutorials to be embedded in documentation using sphinx-gallery

## Getting started

You need **Python 3.6 or greater**.

#### From PyPi

Coming soon!


#### From Github

You can also directly install from the master node (although this is not reccomended)

```
pip install git+https://git@github.com/PyLops/pyproximal.git@master
```

## Contributing

*Feel like contributing to the project? Adding new operators or tutorial?*

We advise using the [Anaconda Python distribution](https://www.anaconda.com/download)
to ensure that all the dependencies are installed via the `Conda` package manager. Follow
the following instructions and read carefully the [CONTRIBUTING](CONTRIBUTING.md)
file before getting started.

### 1. Fork and clone the repository

Execute the following command in your terminal:

```
git clone https://github.com/your_name_here/pyproximal.git
```

### 2. Install PyLops in a new Conda environment
To ensure that further development of PyLops is performed within the same environment (i.e., same dependencies) as
that defined by ``requirements-dev.txt`` or ``environment-dev.yml`` files, we suggest to work off a new Conda enviroment.

The first time you clone the repository run the following command:
```
make dev-install_conda
```
To ensure that everything has been setup correctly, run tests:
```
make tests
```
Make sure no tests fail, this guarantees that the installation has been successfull.

Remember to always activate the conda environment every time you open a new terminal by typing:
```
source activate pyproximal
```

## Documentation
The official documentation of PyProximal is available [here](https://pyproximal.readthedocs.io/).


Moreover, if you have installed PyProximal using the *developer environment*
you can also build the documentation locally by typing the following command:
```
make doc
```
Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing
```
make docupdate
```
Note that if a new example or tutorial is created (and if any change is made to a previously available example or tutorial)
you are required to rebuild the entire documentation before your changes will be visible.


## Contributors
* Matteo Ravasi, mrava87