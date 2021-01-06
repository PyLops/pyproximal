"""
PyProx
======

....
"""
from .ProxOperator import ProxOperator
from .proximal import *

from .utils.utils import Report

from . import proximal
from . import optimization


try:
    from .version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. pylops should be installed
    # properly!
    from datetime import datetime
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')
