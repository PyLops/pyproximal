r"""
Concave penalties
=================
This example considers proximal operators of concave separable penalties.

"""
import matplotlib.pyplot as plt
import numpy as np

import pyproximal

plt.close('all')

###############################################################################
# The SCAD penalty combines both soft-thresholding and hard-thresholding in a
# continuous manner
scad = pyproximal.SCAD(2, 3.7)
x = np.linspace(-10, 10, 101)

fig, ax = plt.subplots(1, 2)
ax[0].plot(x, scad.elementwise(x), label='SCAD')
ax[0].plot(x, np.abs(x), 'k--', label='l1')
ax[0].legend()
ax[1].plot(x, scad.prox(x, 1), label='tau=1')
ax[1].plot(x, scad.prox(x, 0.75), 'b-', label='tau=0.75')
ax[1].plot(x, scad.prox(x, 0.5), 'r-', label='tau=0.50')
ax[1].plot(x, scad.prox(x, 0.25), 'm-', label='tau=0.25')
ax[1].plot(x, x, 'k--')
ax[1].legend()
