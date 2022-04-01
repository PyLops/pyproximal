.. _api:

PyProximal API
==============

The Application Programming Interface (API) of PyProximal is composed of 3 elements:

* *Orthogonal projections*
* *Proximal operators*
* *Solvers*

Orthogonal projections
----------------------

.. currentmodule:: pyproximal.projection

.. autosummary::
   :toctree: generated/

    BoxProj
    HyperPlaneBoxProj
    SimplexProj
    L0BallProj
    L1BallProj
    EuclideanBallProj
    NuclearBallProj
    IntersectionProj
    AffineSetProj

Proximal operators
------------------

Templates
~~~~~~~~~
.. currentmodule:: pyproximal

.. autosummary::
   :toctree: generated/

    ProxOperator

.. currentmodule:: pyproximal.utils.moreau

.. autosummary::
   :toctree: generated/

    moreau


Operators
~~~~~~~~~

.. currentmodule:: pyproximal

.. autosummary::
   :toctree: generated/

    Box
    Simplex
    Intersection
    AffineSet
    Quadratic
    Euclidean
    EuclideanBall
    L0Ball
    L1
    L1Ball
    L2
    L2Convolve
    L21
    L21_plus_L1
    Nonlinear
    Huber
    Nuclear
    NuclearBall
    Orthogonal
    VStack
    SCAD
    Log
    ETP
    Geman


Other operators
---------------

.. currentmodule:: pyproximal.utils.bilinear

.. autosummary::
   :toctree: generated/

    BilinearOperator
    LowRankFactorizedMatrix


Solvers
-------

Primal
~~~~~~

.. currentmodule:: pyproximal.optimization.primal

.. autosummary::
   :toctree: generated/

    ProximalPoint
    ProximalGradient
    AcceleratedProximalGradient
    ADMM
    LinearizedADMM
    TwIST

.. currentmodule:: pyproximal.optimization.sr3

.. autosummary::
   :toctree: generated/

    SR3

.. currentmodule:: pyproximal.optimization.palm

.. autosummary::
   :toctree: generated/

    PALM

.. currentmodule:: pyproximal.optimization.pnp

.. autosummary::
   :toctree: generated/

    PlugAndPlay


Primal-dual
~~~~~~~~~~~

.. currentmodule:: pyproximal.optimization.primaldual

.. autosummary::
   :toctree: generated/

    PrimalDual
    AdaptivePrimalDual

Other
~~~~~

.. currentmodule:: pyproximal.optimization.bregman

.. autosummary::
   :toctree: generated/

    Bregman

.. currentmodule:: pyproximal.optimization.segmentation

.. autosummary::
   :toctree: generated/

    Segment