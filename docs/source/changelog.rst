.. _changlog:

Changelog
=========

Version 0.6.0
--------------
*Released on: 18/03/2023*

|:vertical_traffic_light:| |:vertical_traffic_light:| This is the first release supporting PyLops v2.
|:vertical_traffic_light:| |:vertical_traffic_light:|

* Added ``grad`` method to :py:class:`pyproximal.utils.bilinear.LowRankFactorizedMatrix` operator
* Allow passing optional arguments to solvers in :py:class:`pyproximal.proximal.L2`
* Modified codebase to integrate with pylops's ``cupy`` backend.
* Modified codebase to integrate with ``pylops`` v2.

Version 0.5.0
--------------
*Released on: 20/08/2022*

|:vertical_traffic_light:| |:vertical_traffic_light:| This is the latest release supporting PyLops v1.
|:vertical_traffic_light:| |:vertical_traffic_light:|

* Added :py:class:`pyproximal.proximal.Log1` operator
* Allow ``radius`` parameter of :py:class:`pyproximal.optimization.primal.L0` to be a function
* Allow ``tau`` parameter of :py:class:`pyproximal.optimization.primal.HQS` to be a vector
  and change over iterations
* Added ``z0`` to :py:class:`pyproximal.optimization.primal.HQS`
* Added ``factorize`` option to ``densesolver`` of :py:class:`pyproximal.proximal.L2`

Version 0.4.0
--------------
*Released on: 05/06/2022*

* Added :py:class:`pyproximal.optimization.primal.ADMML2`,
  :py:class:`pyproximal.optimization.primal.HQS`,
  and :py:class:`pyproximal.optimization.pnp.PlugAndPlay` solvers
* Added :py:class:`pyproximal.proximal.ETP`, :py:class:`pyproximal.proximal.Geman`,
  :py:class:`pyproximal.proximal.L0`, :py:class:`pyproximal.proximal.Log`,
  :py:class:`pyproximal.proximal.QuadraticEnvelopeCard`, :py:class:`pyproximal.proximal.SCAD`
  operators.
* Allow ``tau`` parameter of proximal operators to be a vector to handle problems with
  multiple right-hand sides.

Version 0.3.0
--------------
*Released on: 23/03/2022*

* Added :py:class:`pyproximal.optimization.palm.PALM` optimizer
* Added ``callback`` to :py:class:`pyproximal.optimization.primal.ProximalPoint`
  optimizer
* Added :py:class:`pyproximal.utils.bilinear.BilinearOperator`
  and :py:class:`pyproximal.utils.bilinear.LowRankFactorizedMatrix`
  operators

Version 0.2.0
--------------
*Released on: 11/12/2021*

* Added :py:class:`pyproximal.proximal.L0Ball`,
  :py:class:`pyproximal.proximal.L1Ball`,
  :py:class:`pyproximal.proximal.L21_plus_L1`,
  :py:class:`pyproximal.proximal.Nuclear`,
  :py:class:`pyproximal.proximal.NuclearBall`,
  and :py:class:`pyproximal.proximal.Nonlinear` operators
* Added
  :py:class:`pyproximal.proximal.Nuclear`, and
  :py:class:`pyproximal.proximal.NuclearBall` operators
* Added :py:class:`pyproximal.optimization.primal.TwIST` solver
* Added `acceleration` in
  :py:class:`pyproximal.optimization.primal.AcceleratedProximalGradient` solver
* Added classes standard deviation in
  :py:class:`pyproximal.optimization.segmentation.Segment` solver
* Added `chain` method :py:class:`pyproximal.ProxOperator`
* Fix :py:class:`pyproximal.proximal.Orthogonal` by introducing `alpha`
  in the proximal evaluation


Version 0.1.0
--------------
*Released on: 24/04/2021*

* Added :py:class:`pyproximal.optimization.sr3.SR3` solver
* Added :py:class:`pyproximal.projection.AffineSetProj` and
  :py:class:`pyproximal.AffineSet` operators
* Fixed :py:class:`pyproximal.Huber` operator


Version 0.0.0
-------------
*Released on: 17/01/2021*

* First official release.
