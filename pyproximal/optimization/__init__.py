"""
Optimization
============

The subpackage optimization provides an extensive set of proximal algorithms
to be used with PyLops linear operators and PyProximal proximal operators.

A list of solvers in ``pylops.optimization.proximal`` using only proximal
operators:

    ProximalPoint                   Proximal point algorithm (or proximal min.)
    ProximalGradient                Proximal gradient algorithm
    AcceleratedProximalGradient     Accelerated Proximal gradient algorithm
    ADMM                            Alternating Direction Method of Multipliers
    LinearizedADMM                  Linearized ADMM

A list of solvers in ``pylops.optimization.proximaldual`` using both proximal
and dual proximal operators:

    PrimalDual                      Primal-Dual algorithm
    AdaptivePrimalDual              Adaptive Primal-Dual algorithm

and an higher level solvers based on Bregman iterations (which can be combined
with any of the above solvers to solve each subproblem in its inner loop)

    Bregman                         Bregman iterations

Finally this subpackage contains also a solver for image segmentation based
on a special use of the Primal-Dual algorithm:

    Segment                         Image Segmentation via Primal-Dual algorithm

"""

from . import primal, primaldual, bregman, segmentation