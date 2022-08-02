"""
Proximal Operators
==================

The subpackage proximal contains a number of proximal operators:

    Box                             Box indicator
    Simplex                         Simplex indicator
    Intersection                    Intersection indicator
    AffineSet	                      Affines set indicator
    Quadratic                       Quadratic function
    Nonlinear	                      Nonlinear function
    L0                              L0 Norm
    L0Ball                          L0 Ball
    L1	                            L1 Norm
    L1Ball                          L1 Ball
    Euclidean	                      Euclidean Norm
    EuclideanBall	                  Euclidean Ball
    L2	                            L2 Norm
    L2Convolve	                    L2 Norm of convolution operator
    L21	                            L2,1 Norm
    L21_plus_L1	                    L2,1 + L1 mixed-norm
    Huber	                          Huber Norm
    TV                              Total Variation Norm                                   
    Nuclear                         Nuclear Norm
    NuclearBall                     Nuclear Ball
    Orthogonal	                    Product between orthogonal operator and vector
    VStack	                        Stack of proximal operators
    SCAD                            Smoothly clipped absolute deviation
    Log                             Logarithmic
    Log1                            Another form of logarithmic
    ETP                             Exponential-type penalty
    Geman                           Geman penalty
    QuadraticEnvelopeCard           The quadratic envelope of the cardinality function
    SingularValuePenalty            Generic singular value penalty
    QuadraticEnvelopeCardIndicator  The quadratic envelope of the indicator function of the cardinality function
    Hankel                          Hankel indicator
    QuadraticEnvelopeRankL2         The quadratic envelope of the rank function with an L2 misfit term

"""

from .Box import *
from .Simplex import *
from .Intersection import *
from .AffineSet import *
from .Quadratic import *
from .Nonlinear import *
from .Euclidean import *
from .L0 import *
from .L1 import *
from .L2 import *
from .L21 import *
from .L21_plus_L1 import *
from .Huber import *
from .TV import *
from .Nuclear import *
from .Orthogonal import *
from .VStack import *
from .SCAD import *
from .Log import *
from .ETP import *
from .Geman import *
from .QuadraticEnvelope import *
from .SingularValuePenalty import *
from .Hankel import *

__all__ = ['Box', 'Simplex', 'Intersection', 'AffineSet', 'Quadratic',
           'Euclidean', 'EuclideanBall', 'L0', 'L0Ball', 'L1', 'L1Ball', 'L2',
           'L2Convolve', 'L21', 'L21_plus_L1', 'Huber', 'TV', 'Nuclear',
           'NuclearBall', 'Orthogonal', 'VStack', 'Nonlinear', 'SCAD',
           'Log', 'Log1', 'ETP', 'Geman', 'QuadraticEnvelopeCard', 'SingularValuePenalty',
           'QuadraticEnvelopeCardIndicator', 'QuadraticEnvelopeRankL2',
           'Hankel']
