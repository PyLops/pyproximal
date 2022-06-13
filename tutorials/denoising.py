r"""
Denoising
=========
This tutorial considers the classical problem of denoising of images affected
by either random noise or salt-and-pepper noise using proximal algorithms.

The overall cost function to minimize is written in the following form:

    .. math::
        \argmin_\mathbf{u} \frac{1}{2}\|\mathbf{u}-\mathbf{f}\|_2^2 +
        \sigma J(\mathbf{u})

where the L2 norm in the data term can be replaced by a L1 norm for
salt-and-pepper (outlier like noise).

For both examples we investigate with different choices of regularization:

- L2 on Gradient :math:`J(\mathbf{u}) = \|\nabla \mathbf{u}\|_2^2`
- Anisotropic TV :math:`J(\mathbf{u}) = \|\nabla \mathbf{u}\|_1`
- Isotropic TV :math:`J(\mathbf{u}) = \|\nabla \mathbf{u}\|_{2,1}`

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops
from scipy import misc

import pyproximal

plt.close('all')

###############################################################################
# Let's start by loading a sample image and adding some noise

# Load image
img = misc.ascent()
img = img / np.max(img)
ny, nx = img.shape

# Add noise
sigman = .2
n = sigman * np.max(abs(img.ravel())) * np.random.uniform(-1, 1, img.shape)
noise_img = img + n

###############################################################################
# We can now define a :class:`pylops.Gradient` operator that we are going to
# use for all regularizers

# Gradient operator
sampling = 1.
Gop = pylops.Gradient(dims=(ny, nx), sampling=sampling, edge=False,
                      kind='forward', dtype='float64')
L = 8. / sampling ** 2 # maxeig(Gop^H Gop)

###############################################################################
# We then consider the first regularization (L2 norm on Gradient). We expect
# to get a smooth image where noise is suppressed by sharp edges in the
# original image are however lost.

# L2 data term
l2 = pyproximal.L2(b=noise_img.ravel())

# L2 regularization
sigma = 2.
thik = pyproximal.L2(sigma=sigma)

# Solve
tau = 1.
mu = 1. / (tau*L)

iml2 = pyproximal.optimization.primal.LinearizedADMM(l2, thik,
                                                     Gop, tau=tau,
                                                     mu=mu,
                                                     x0=np.zeros_like(img.ravel()),
                                                     niter=100)[0]
iml2 = iml2.reshape(img.shape)

###############################################################################
# Let's try now to use TV regularization, both anisotropic and isotropic

# L2 data term
l2 = pyproximal.L2(b=noise_img.ravel())

# Anisotropic TV
sigma = .1
l1 = pyproximal.L1(sigma=sigma)

# Solve
tau = 1.
mu = tau / L

iml1 = pyproximal.optimization.primal.LinearizedADMM(l2, l1, Gop, tau=tau,
                                                     mu=mu, x0=np.zeros_like(img.ravel()),
                                                     niter=100)[0]
iml1 = iml1.reshape(img.shape)


# Isotropic TV with Proximal Gradient
sigma = .1
tv = pyproximal.TV(dims=img.shape, sigma=sigma)

# Solve
tau = 1 / L

imtv = pyproximal.optimization.primal.ProximalGradient(l2, tv, tau=tau, x0=np.zeros_like(img.ravel()),
                                                       niter=100)
imtv = imtv.reshape(img.shape)

# Isotropic TV with Primal Dual
sigma = .1
l1iso = pyproximal.L21(ndim=2, sigma=sigma)

# Solve
tau = 1 / np.sqrt(L)
mu = 1. / (tau*L)

iml12 = pyproximal.optimization.primaldual.PrimalDual(l2, l1iso, Gop,
                                                      tau=tau, mu=mu, theta=1.,
                                                      x0=np.zeros_like(img.ravel()),
                                                      niter=100)
iml12 = iml12.reshape(img.shape)

fig, axs = plt.subplots(1, 5, figsize=(14, 4))
axs[0].imshow(img, cmap='gray', vmin=0, vmax=1)
axs[0].set_title('Original')
axs[0].axis('off')
axs[0].axis('tight')
axs[1].imshow(noise_img, cmap='gray', vmin=0, vmax=1)
axs[1].set_title('Noisy')
axs[1].axis('off')
axs[1].axis('tight')
axs[2].imshow(iml1, cmap='gray', vmin=0, vmax=1)
axs[2].set_title('TVaniso')
axs[2].axis('off')
axs[2].axis('tight')
axs[3].imshow(imtv, cmap='gray', vmin=0, vmax=1)
axs[3].set_title('TViso (with ProxGrad)')
axs[3].axis('off')
axs[3].axis('tight')
axs[4].imshow(iml12, cmap='gray', vmin=0, vmax=1)
axs[4].set_title('TViso (with PD)')
axs[4].axis('off')
axs[4].axis('tight')
plt.tight_layout()

###############################################################################
# Finally we consider an example where the original image is corrupted by
# salt-and-pepper noise.

# Add salt and pepper noise
noiseperc = .1

isalt = np.random.permutation(np.arange(ny*nx))[:int(noiseperc*ny*nx)]
ipepper = np.random.permutation(np.arange(ny*nx))[:int(noiseperc*ny*nx)]
noise_img = img.copy().ravel()
noise_img[isalt] = img.max()
noise_img[ipepper] = img.min()
noise_img = noise_img.reshape(ny, nx)

###############################################################################
# Here we compare L2 and L1 norms for the data term
# L2 data term
l2 = pyproximal.L2(b=noise_img.ravel())

# L1 regularization (isotropic TV)
sigma = .2
l1iso = pyproximal.L21(ndim=2, sigma=sigma)

# Solve
tau = .1
mu = 1. / (tau*L)

iml12_l2 = pyproximal.optimization.primaldual.PrimalDual(l2, l1iso, Gop,
                                                         tau=tau, mu=mu, theta=1.,
                                                         x0=np.zeros_like(noise_img).ravel(),
                                                         niter=100, show=True)
iml12_l2 = iml12_l2.reshape(img.shape)


# L1 data term
l1 = pyproximal.L1(g=noise_img.ravel())

# L1 regularization (isotropic TV)
sigma = .7
l1iso = pyproximal.L21(ndim=2, sigma=sigma)

# Solve
tau = 1.
mu = 1. / (tau*L)

iml12_l1 = pyproximal.optimization.primaldual.PrimalDual(l1, l1iso, Gop,
                                                         tau=tau, mu=mu, theta=1.,
                                                         x0=np.zeros_like(noise_img).ravel(),
                                                         niter=100, show=True)
iml12_l1 = iml12_l1.reshape(img.shape)

fig, axs = plt.subplots(2, 2, figsize=(14, 14))
axs[0][0].imshow(img, cmap='gray', vmin=0, vmax=1)
axs[0][0].set_title('Original')
axs[0][0].axis('off')
axs[0][0].axis('tight')
axs[0][1].imshow(noise_img, cmap='gray', vmin=0, vmax=1)
axs[0][1].set_title('Noisy')
axs[0][1].axis('off')
axs[0][1].axis('tight')
axs[1][0].imshow(iml12_l2, cmap='gray', vmin=0, vmax=1)
axs[1][0].set_title('L2data + TViso')
axs[1][0].axis('off')
axs[1][0].axis('tight')
axs[1][1].imshow(iml12_l1, cmap='gray', vmin=0, vmax=1)
axs[1][1].set_title('L1data + TViso')
axs[1][1].axis('off')
axs[1][1].axis('tight')
plt.tight_layout()