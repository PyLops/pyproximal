# 0.3.0
* Added ``pyproximal.optimization.palm.PALM`` optimizer
* Added ``callback`` to ``pyproximal.optimization.proximal.ProximalPoint`` 
  optimizer
* Added ``pyproximal.utils.bilinear.BilinearOperator`` and 
  ``pyproximal.utils.bilinear.LowRankFactorizedMatrix`` operators

# 0.2.0
* Added ``pyproximal.proximal.L0Ball``, ``pyproximal.proximal.L1Ball``, 
  ``pyproximal.proximal.L21_plus_L1``, ``pyproximal.proximal.Nuclear``, 
  ``pyproximal.proximal.NuclearBall``, and ``pyproximal.proximal.Nonlinear`` 
  operators
* Added ``pyproximal.optimization.primal.TwIST`` solver
* Added `acceleration` in
  ``pyproximal.optimization.primal.AcceleratedProximalGradient`` solver
* Added classes standard deviation in
  ``pyproximal.optimization.segmentation.Segment`` solver
* Added `chain` method ``pyproximal.ProxOperator``
* Fix ``pyproximal.proximal.Orthogonal`` by introducing `alpha`
  in the proximal evaluation
  
# 0.1.0
* Added ``pyproximal.optimization.sr3.SR3`` solver
* Added ``pyproximal.projection.AffineSetProj`` and
  ``pyproximal.AffineSet`` operators
* Fixed ``pyproximal.Huber`` operator

# 0.0.0
* First official release.

