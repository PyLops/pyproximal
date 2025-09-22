
r"""
Dykstra's algorithms
==============================================

This example showcases two closely related tasks:
projection onto an intersection of convex sets
using the Dykstra's projection algorithm,
and proximal operator of a sum of proximable functions
suing the Dykstra-like proximal algorithm.

"""

###############################################################################
# Here is an example of a projection onto the intersection of convex sets
# using :class:`pyproximal.projection.GenericIntersectionProj`.

import numpy as np
from pyproximal.projection import (
    BoxProj,
    EuclideanBallProj,
    GenericIntersectionProj
)

circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_proj = GenericIntersectionProj(projections)

rng = np.random.default_rng(10)
x = rng.normal(0., 3.5, size=2)
print("x            =", x)

xp = dykstra_proj(x)

print("x projection =", xp)

###############################################################################
# Let's see how x is projected to xp.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import to_rgba

fig, ax = plt.subplots(figsize=(6, 6))

circles = [
    ((-2.5, 0.0), 5.0),
    ((2.5, 0.0), 5.0),
    ((0.0, 3.5), 5.0),
]
for (cx, cy), r in circles:
    ax.add_patch(Circle(
        (cx, cy), r,
        facecolor=to_rgba('C0', 0.06),
        edgecolor='k', linewidth=0.5, linestyle='-'
    ))

xmin, ymin = (-5.0, -2.5)
xmax, ymax = (5.0, 2.5)
ax.add_patch(Rectangle(
    (xmin, ymin), xmax - xmin, ymax - ymin,
    facecolor=to_rgba('C1', 0.06),
    edgecolor='k', linewidth=0.5, linestyle='-',
))

ax.scatter(*x, marker='o', s=40, color='k', label="x")
ax.scatter(*xp, marker='o', s=40, color='red', label="xp")
ax.annotate('', xy=xp, xytext=x, arrowprops=dict(arrowstyle='->', color='k'))

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.grid(alpha=0.2)
ax.legend()

plt.show()

###############################################################################
# Here is another example of the same projection onto the intersection of convex sets
# using :class:`pyproximal.GenericIntersectionProx`.

import numpy as np
from pyproximal.projection import (
    BoxProj,
    EuclideanBallProj,
)
from pyproximal.proximal import GenericIntersectionProx

# projection functions
circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_prox = GenericIntersectionProx(projections)

rng = np.random.default_rng(10)
x = rng.normal(0., 3.5, size=2)

print("x            =", x)
print("Is x inside?", dykstra_prox(x))  # x is outside

xp = dykstra_prox.prox(x, 1.0)

print("x projection =", xp)
print("Is x inside?", dykstra_prox(xp))  # xp is inside

###############################################################################
# Yet another example of the same projection onto the intersection of convex sets
# using :class:`pyproximal.Sum` via the sum of indicator functions of the projections.

import numpy as np
from pyproximal.proximal import (
    Box,
    EuclideanBall,
    Sum,
)

# indicator functions
circle_1 = EuclideanBall(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBall(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBall(np.array([0.0, 3.5]), 5)
box = Box(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_sum = Sum(projections)  # sum of indicator functions

rng = np.random.default_rng(10)
x = rng.normal(0., 3.5, size=2)

print("x            =", x)
print("Is x inside?", dykstra_sum(x))  # x is outside

xp = dykstra_sum.prox(x, 1.0)

print("x projection =", xp)
print("Is x inside?", dykstra_sum(xp))  # xp should be inside, but round-off error can leave it marginally infeasible.

###############################################################################
# Here is an example of computing proximal operator of the sum of proximable functions
# using :class:`pyproximal.Sum`.

import numpy as np
from pyproximal.proximal import L1, L2, Box, Sum
from pylops import MatrixMult
rng = np.random.default_rng(10)

A = MatrixMult(rng.normal(0., 1., size=(3, 5)))
b = rng.normal(0., 1., size=3)
sigma = rng.normal(0., 1.)
l2_term = L2(A, b)
l1_term = L1(sigma=sigma)
box = Box(rng.uniform(-5, -2.5, size=5), rng.uniform(2.5, 5, size=5))

# for computing prox of 1/2 * ||Ax - b||_2^2 + sigma ||x||_1 + I_box(x)
dykstra = Sum([l2_term, l1_term, box])

x = rng.normal(0., 5., size=5)
tau = 1.0

prox_x = dykstra.prox(x, tau)

print("x      =", x)
print("prox(x)=", prox_x)
