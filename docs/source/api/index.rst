.. _api:

PyProximal API
==============

The Application Programming Interface (API) of PyProx...


Orthogonal projections
----------------------

.. currentmodule:: pyproximal.projection

.. autosummary::
   :toctree: generated/

    BoxProj
    HyperPlaneBoxProj
    SimplexProj
    EuclideanBallProj
    IntersectionProj


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
    Quadratic
    Euclidean
    EuclideanBall
    L1
    L2
    L21
    Huber
    Orthogonal
    L2Convolve
    VStack


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