"""
Proximal Operators
==================

The subpackage proximal contains a number of proximal operators:

    Box                             Box Indicator
    Quadratic                       Quadratic function
    Euclidean	                    Euclidean Norm
    L1	                            L1 Norm
    L2	                            L2 Norm

"""

from .Box import *
from .Simplex import *
from .Intersection import *
from .Quadratic import *
from .Euclidean import *
from .L1 import *
from .L2 import *
from .L21 import *
from .Huber import *
from .Orthogonal import *
from .VStack import *


__all__ = ['Box', 'Simplex', 'Intersection', 'Quadratic', 'Euclidean',
           'EuclideanBall', 'L1', 'L2', 'L2Convolve', 'L21', 'Huber',
           'Orthogonal', 'VStack']