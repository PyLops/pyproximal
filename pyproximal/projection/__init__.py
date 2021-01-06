"""
Orthogonal Projections
======================

The subpackage projection contains a number of orthogonal projection:

    Box	                    Projection onto a Box

"""

from .Box import *
from .Simplex import *
from .Euclidean import *
from .Intersection import *


__all__ = ['BoxProj', 'HyperPlaneBoxProj', 'SimplexProj', 'EuclideanBallProj',
           'IntersectionProj']