"""
PyProximal
==========

This Python library provides all the needed building blocks for solving
non-smooth convex optimization problems using the so-called proximal algorithms.

PyProximal provides
  1. A general construct for creating proximal operators
  2. An extensive set of commonly used proximal operators
  3. A set of solvers for composite objective functions with
     differentiable and proximable functions.

Available subpackages
---------------------
projection
    Project Operators
proximal
    Proximal Operators
optimization
    Solvers
utils
    Utility routines

"""

import logging

from .ProxOperator import ProxOperator
from .proximal import *

from .utils.utils import Report

from . import proximal
from . import optimization

# Prevent no handler message if an application using PyProximal does not configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. PyProximal should be installed
    # properly!
    from datetime import datetime

    __version__ = "unknown-" + datetime.today().strftime("%Y%m%d")
