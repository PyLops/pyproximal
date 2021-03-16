"""
Proximal Operators
==================

The subpackage proximal contains a number of proximal operators:

    Box                             Box indicator
    Simplex                         Simplex indicator
    Intersection                    Intersection indicator
    AffineSet	                    Affines set indicator
    Quadratic                       Quadratic function
    Euclidean	                    Euclidean Norm
    EuclideanBall	                Euclidean Ball
    L1	                            L1 Norm
    L2	                            L2 Norm
    L2Convolve	                    L2 Norm of convolution operator
    L21	                            L2,1 Norm
    Huber	                        Huber norm
    Orthogonal	                    Product between orthogonal operator and vector
    VStack	                        Stack of proximal operators

"""

from .Box import *
from .Simplex import *
from .Intersection import *
from .AffineSet import *
from .Quadratic import *
from .Euclidean import *
from .L1 import *
from .L2 import *
from .L21 import *
from .Huber import *
from .Orthogonal import *
from .VStack import *


__all__ = ['Box', 'Simplex', 'Intersection', 'AffineSet', 'Quadratic',
           'Euclidean', 'EuclideanBall', 'L1', 'L2', 'L2Convolve', 'L21',
           'Huber', 'Orthogonal', 'VStack']
