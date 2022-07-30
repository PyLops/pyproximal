"""
Orthogonal Projections
======================

The subpackage projection contains a number of orthogonal projection:

    BoxProj	                    Projection onto a Box
    HyperPlaneBoxProj	        Projection onto an intersection beween a HyperPlane and a Box
    SimplexProj	                Projection onto a Simplex
    L0Proj	                    Projection onto an L0 Ball
    L1Proj	                    Projection onto an L1 Ball
    EuclideanBallProj	        Projection onto an Euclidean Ball
    NuclearBallProj	            Projection onto a Nuclear Ball
    IntersectionProj	        Projection onto an Intersection of sets
    AffineSetProj	            Projection onto an Affine set
    HankelProj                  Projection onto the set of Hankel matrices

"""

from .Box import *
from .Simplex import *
from .L0 import *
from .L1 import *
from .Euclidean import *
from .Nuclear import *
from .Intersection import *
from .AffineSet import *
from .Hankel import *


__all__ = ['BoxProj', 'HyperPlaneBoxProj', 'SimplexProj', 'L0BallProj',
           'L1BallProj', 'EuclideanBallProj', 'NuclearBallProj',
           'IntersectionProj', 'AffineSetProj', 'HankelProj']