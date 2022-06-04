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
    HQS                             Half Quadrating Splitting
    ADMM                            Alternating Direction Method of Multipliers
    ADMML2                          ADMM with L2 misfit term
    LinearizedADMM                  Linearized ADMM
    TwIST                           Two-step Iterative Shrinkage/Threshold
    PlugAndPlay                     Plug-and-Play Prior with ADMM

A list of solvers in ``pylops.optimization.proximaldual`` using both proximal
and dual proximal operators:

    PrimalDual                      Primal-Dual algorithm
    AdaptivePrimalDual              Adaptive Primal-Dual algorithm

and an higher level solvers based on Bregman iterations (which can be combined
with any of the above solvers to solve each subproblem in its inner loop)

    Bregman                         Bregman iterations

Additional solvers are in ``pylops.optimization.sr3`` amd
``pylops.optimization.palm``:

    SR3                             Sparse Relaxed Regularized algorithm
    PALM                            Proximal Alternating Linearized Minimization

Finally this subpackage contains also a solver for image segmentation based
on a special use of the Primal-Dual algorithm:

    Segment                         Image Segmentation via Primal-Dual algorithm

"""

from . import primal, primaldual, bregman, segmentation, sr3, palm, pnp