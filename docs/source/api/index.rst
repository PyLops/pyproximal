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

    AffineSetProj
    BoxProj
    EuclideanBallProj
    HyperPlaneBoxProj
    IntersectionProj
    L0BallProj
    L1BallProj
    NuclearBallProj
    SimplexProj

Proximal operators
------------------

Templates
^^^^^^^^^

.. currentmodule:: pyproximal

.. autosummary::
   :toctree: generated/

    ProxOperator

.. currentmodule:: pyproximal.utils.moreau

.. autosummary::
   :toctree: generated/

    moreau

.. currentmodule:: pyproximal

Vector
^^^^^^

Convex
~~~~~~

.. autosummary::
   :toctree: generated/

    AffineSet
    Box
    Euclidean
    EuclideanBall
    Huber
    Intersection
    L0
    L0Ball
    L1
    L1Ball
    L2
    L2Convolve
    L21
    L21_plus_L1
    Orthogonal
    Quadratic
    Simplex


Non-Convex
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

    ETP
    Geman
    Log
    QuadraticEnvelopeCard
    SCAD


Matrix-only
^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

    Nuclear
    NuclearBall
    SingularValuePenalty


Other
^^^^^

.. autosummary::
   :toctree: generated/

    Nonlinear
    VStack


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
^^^^^^

.. currentmodule:: pyproximal.optimization.primal

.. autosummary::
   :toctree: generated/

    AcceleratedProximalGradient
    ADMM
    ADMML2
    HQS
    LinearizedADMM
    ProximalGradient
    ProximalPoint
    TwIST

.. currentmodule:: pyproximal.optimization.palm

.. autosummary::
   :toctree: generated/

    PALM

.. currentmodule:: pyproximal.optimization.pnp

.. autosummary::
   :toctree: generated/

    PlugAndPlay

.. currentmodule:: pyproximal.optimization.sr3

.. autosummary::
   :toctree: generated/

    SR3


Primal-dual
^^^^^^^^^^^

.. currentmodule:: pyproximal.optimization.primaldual

.. autosummary::
   :toctree: generated/

    AdaptivePrimalDual
    PrimalDual

Other
^^^^^

.. currentmodule:: pyproximal.optimization.bregman

.. autosummary::
   :toctree: generated/

    Bregman

.. currentmodule:: pyproximal.optimization.segmentation

.. autosummary::
   :toctree: generated/

    Segment