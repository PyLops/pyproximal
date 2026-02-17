r"""
Dykstra's algorithms
====================
This example showcases two closely related tasks:

- projection onto an intersection of convex sets using the
  Dykstra's projection algorithm;
- proximal operator of a sum of proximable functions using
  the Dykstra-like proximal algorithm.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
from pylops import MatrixMult
from pyproximal.projection import BoxProj, EuclideanBallProj, GenericIntersectionProj
from pyproximal.proximal import L1, L2, Box, EuclideanBall, GenericIntersectionProx, Sum

rng = np.random.default_rng(10)

###############################################################################
# Here is an example of a projection onto the intersection of convex sets
# using :class:`pyproximal.projection.GenericIntersectionProj`.

circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_proj = GenericIntersectionProj(projections)

x = rng.normal(-5.0, 1.5, size=2)
print("x            =", x)

xp = dykstra_proj(x)

print("x projection =", xp)

###############################################################################
# Let's now see how :math:`\mathbf{x}` is projected to :math:`\mathbf{x_p}`.

fig, ax = plt.subplots(figsize=(6, 6))

circles = [
    ((-2.5, 0.0), 5.0),
    ((2.5, 0.0), 5.0),
    ((0.0, 3.5), 5.0),
]
for (cx, cy), r in circles:
    ax.add_patch(
        Circle(
            (cx, cy),
            r,
            facecolor=to_rgba("C0", 0.06),
            edgecolor="k",
            linewidth=0.5,
            linestyle="-",
        )
    )

xmin, ymin = (-5.0, -2.5)
xmax, ymax = (5.0, 2.5)
ax.add_patch(
    Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        facecolor=to_rgba("C1", 0.06),
        edgecolor="k",
        linewidth=0.5,
        linestyle="-",
    )
)

ax.scatter(x[0], x[1], s=40, c="k", marker="o", label="x")
ax.scatter(xp[0], xp[1], s=40, c="red", marker="o", label="xp")
ax.annotate(
    "",
    xy=(xp[0], xp[1]),
    xytext=(x[0], x[1]),
    arrowprops={"arrowstyle": "->", "color": "k"},
)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.grid(alpha=0.2)
ax.legend()

plt.show()

###############################################################################
# Similarly, we can use the :class:`pyproximal.GenericIntersectionProx` to
# perform the same projection onto the intersection of convex sets; this is
# usually the preferred choice when we want to pass proximal operators of
# indicator functions to any of our proximal solvers.

# projection functions
circle_1 = EuclideanBallProj(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBallProj(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBallProj(np.array([0.0, 3.5]), 5)
box = BoxProj(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_prox = GenericIntersectionProx(projections)

x = rng.normal(0.0, 3.5, size=2)

print("x            =", x)
print("Is x inside?", dykstra_prox(x))  # x is outside

xp = dykstra_prox.prox(x, 1.0)

print("x projection =", xp)
print("Is x inside?", dykstra_prox(xp))  # xp is inside

###############################################################################
# Note that with an abuse of notation, the same projection
# can also be performed using :class:`pyproximal.Sum` (i.e., the sum of indicator
# functions of the projections). Whilst this is possible, we reccomend using
# :class:`pyproximal.GenericIntersectionProx` when dealing with indicator
# functions for improved code clarity.

# indicator functions
circle_1 = EuclideanBall(np.array([-2.5, 0.0]), 5)
circle_2 = EuclideanBall(np.array([2.5, 0.0]), 5)
circle_3 = EuclideanBall(np.array([0.0, 3.5]), 5)
box = Box(np.array([-5.0, -2.5]), np.array([5.0, 2.5]))

projections = [circle_1, circle_2, circle_3, box]
dykstra_sum = Sum(projections)  # sum of indicator functions

x = rng.normal(0.0, 3.5, size=2)

print("x            =", x)
print("Is x inside?", dykstra_sum(x))  # x is outside

xp = dykstra_sum.prox(x, 1.0)

print("x projection =", xp)
print(
    "Is x inside?", dykstra_sum(xp)
)  # note that round-off error may leave it marginally infeasible.

###############################################################################
# Finally, let's use :class:`pyproximal.Sum` in the correct way, i.e. with
# proximable functions. This will compute the proximal operator of the sum of the
# functions we have passed. Note that the reason why we have shown
# :class:`pyproximal.GenericIntersectionProx` and :class:`pyproximal.Sum` is
# that under the hood they both rely on similar versions of the Dykstra algorithm.

A = MatrixMult(rng.normal(0.0, 1.0, size=(3, 5)))
b = rng.normal(0.0, 1.0, size=3)
sigma = rng.normal(0.0, 1.0)
l2_term = L2(A, b)
l1_term = L1(sigma=sigma)
box = Box(rng.uniform(-5, -2.5, size=5), rng.uniform(2.5, 5, size=5))

# for computing prox of 1/2 * ||Ax - b||_2^2 + sigma ||x||_1 + I_box(x)
dykstra = Sum([l2_term, l1_term, box])

x = rng.normal(0.0, 5.0, size=5)
tau = 1.0

prox_x = dykstra.prox(x, tau)

print("x      =", x)
print("prox(x)=", prox_x)
