"""
Orthogonal Projections
======================

The subpackage projection contains a number of orthogonal projection:

    BoxProj	                    Projection onto a Box
    HyperPlaneBoxProj	        Projection onto an intersection beween a HyperPlane and a Box
    SimplexProj	                Projection onto a Simplex
    EuclideanBallProj	        Projection onto an Euclidean Ball
    L1Proj	                    Projection onto an L1 Ball
    NuclearBallProj	            Projection onto a Nuclear Ball
    IntersectionProj	        Projection onto an Intersection of sets
    AffineSetProj	            Projection onto an Affine set

"""

from .Box import *
from .Simplex import *
from .Euclidean import *
from .L1 import *
from .Nuclear import *
from .Intersection import *
from .AffineSet import *


__all__ = ['BoxProj', 'HyperPlaneBoxProj', 'SimplexProj', 'EuclideanBallProj',
           'L1BallProj', 'NuclearBallProj', 'IntersectionProj', 'AffineSetProj']