r"""
Low-Rank completion via SVD
===========================
In this tutorial we will present an example of low-rank matrix completion.
Contrarily to most of the examples in this library (and PyLops), the regularizer
is here applied to a matrix, which is obtained by reshaping the model vector
that we wish to solve for.

In this example we will consider the following forward problem:

.. math::
    \mathbf{y} = \mathbf{R} \mathbf{x}

where :math:`\mathbf{R}` is a restriction operator, which applied to
:math:`\mathbf{x}=\operatorname{vec}(\mathbf{X})`, the vectorized version of a 2d image of
size :math:`n \times m`, selects a reasonably small number of samples
:math:`p \ll nm` that form the vector :math:`\mathbf{y}`. Note that any other
modelling operator could be used here, for example a 2D convolutional operator
in the case of deblurring or a 2D FFT plus restriction in the case of
MRI scanning.

The problem we want to solve can be mathematically described as:

.. math::
    \argmin_\mathbf{x} \frac{1}{2}\|\mathbf{y}-\mathbf{Rx}\|_2^2 + \mu \|\mathbf{X}\|_*

or

.. math::
    \argmin_\mathbf{x} \frac{1}{2}\|\mathbf{y}-\mathbf{Rx}\|_2^2 \; \text{s.t.}
    \; \|\mathbf{X}\|_* < \mu

where :math:`\|\mathbf{X}\|_*=\sum_i \sigma_i` is the nuclear norm of
:math:`\mathbf{X}` (i.e., the sum of the singular values).

"""
# sphinx_gallery_thumbnail_number = 2
import numpy as np
import matplotlib.pyplot as plt
import pylops
import pyproximal

from scipy import misc

np.random.seed(0)
plt.close('all')

###############################################################################
# Let's start by loading a sample image

# Load image
X = misc.ascent()
X = X/np.max(X)
ny, nx = X.shape

###############################################################################
# We can now define a :class:`pylops.Restriction` operator and look at how
# the singular values of our image change when we remove some of its sample.

# Restriction operator
sub = 0.4
nsub = int(ny*nx*sub)
iava = np.random.permutation(np.arange(ny*nx))[:nsub]

Rop = pylops.Restriction(ny*nx, iava)

# Data
y = Rop * X.ravel()

# Masked data
Y = (Rop.H * Rop * X.ravel()).reshape(ny, nx)

# SVD of true and masked data
Ux, Sx, Vhx = np.linalg.svd(X, full_matrices=False)
Uy, Sy, Vhy = np.linalg.svd(Y, full_matrices=False)

plt.figure()
plt.semilogy(Sx, 'k', label=r'$||X||_*$=%.2f' % np.sum(Sx))
plt.semilogy(Sy, 'r', label=r'$||Y||_*$=%.2f' % np.sum(Sy))
plt.legend()
plt.tight_layout()

###############################################################################
# We observe that removing some samples from the image has led to an overall
# increase in the singular values of :math:`\mathbf{X}`, especially
# those that are originally very small. As a consequence the nuclear norm of
# :math:`\mathbf{Y}` (the masked image) is larger than that of
# :math:`\mathbf{X}`.
#
# Let's now set up the inverse problem using the Proximal gradient algorithm

mu = .8
f = pyproximal.L2(Rop, y)
g = pyproximal.Nuclear((ny, nx), mu)

Xpg = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                      tau=1., niter=100, show=True)
Xpg = Xpg.reshape(ny, nx)

# Recompute SVD and see how the singular values look like
Upg, Spg, Vhpg = np.linalg.svd(Xpg, full_matrices=False)

###############################################################################
# Let's do the same with the constrained version
mu1 = 0.8 * np.sum(Sx)
g = pyproximal.proximal.NuclearBall((ny, nx), mu1)

Xpgc = pyproximal.optimization.primal.ProximalGradient(f, g, np.zeros(ny*nx), acceleration='vandenberghe',
                                                       tau=1., niter=100, show=True)
Xpgc = Xpgc.reshape(ny, nx)

# Recompute SVD and see how the singular values look like
Upgc, Spgc, Vhpgc = np.linalg.svd(Xpgc, full_matrices=False)

###############################################################################
# And finally we display the reconstructed image

plt.figure()
plt.semilogy(Sx, 'k', label=r'$||X||_*$=%.2f' % np.sum(Sx))
plt.semilogy(Sy, 'r', label=r'$||Y||_*$=%.2f' % np.sum(Sy))
plt.semilogy(Spg, 'b', label=r'$||X_{pg}||_*$=%.2f' % np.sum(Spg))
plt.semilogy(Spgc, 'g', label=r'$||X_{pgc}||_*$=%.2f' % np.sum(Spgc))
plt.legend()
plt.tight_layout()

fig, axs = plt.subplots(1, 4, figsize=(14, 6))
axs[0].imshow(X, cmap='gray')
axs[0].set_title('True')
axs[1].imshow(Y, cmap='gray')
axs[1].set_title('Masked')
axs[2].imshow(Xpg, cmap='gray')
axs[2].set_title('Reconstructed reg.')
axs[3].imshow(Xpgc, cmap='gray')
axs[3].set_title('Reconstructed constr.')
fig.tight_layout()
