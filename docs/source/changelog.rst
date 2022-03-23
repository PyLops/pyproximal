.. _changlog:

Changelog
=========

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
  :py:class:`pyproximal.proximal.NuclearBall` operator
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
