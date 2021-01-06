r"""
Adaptive Primal-Dual
====================
This tutorial compares the traditional Chambolle-Pock Primal-dual algorithm
with the Adaptive Primal-Dual Hybrid Gradient of Goldstein and co-authors.

By adaptively changing the step size in the primal and the dual directions,
this algorithm shows faster convergence, which is of great importance for some
of the problems that the Primal-Dual algorithm can solve - especially those
with an expensive proximal operator.

For this example, we consider a simple denoising problem.

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops
from skimage.data import camera

import pyproximal

plt.close('all')

def callback(x, f, g, K, cost, xtrue, err):
    cost.append(f(x) + g(K.matvec(x)))
    err.append(np.linalg.norm(x - xtrue))

###############################################################################
# Let's start by loading a sample image and adding some noise

# Load image
img = camera()
ny, nx = img.shape

# Add noise
sigman = 20
n = np.random.normal(0, sigman, img.shape)
noise_img = img + n

###############################################################################
# We can now define a :class:`pylops.Gradient` operator as well as the
# different proximal operators to be passed to our solvers

# Gradient operator
sampling = 1.
Gop = pylops.Gradient(dims=(ny, nx), sampling=sampling, edge=False,
                      kind='forward', dtype='float64')
L = 8. / sampling ** 2 # maxeig(Gop^H Gop)

# L2 data term
lamda = .04
l2 = pyproximal.L2(b=noise_img.ravel(), sigma=lamda)

# L1 regularization (isotropic TV)
l1iso = pyproximal.L21(ndim=2)

###############################################################################
# To start, we solve our denoising problem with the original Primal-Dual
# algorithm

# Primal-dual
tau = 0.95 / np.sqrt(L)
mu = 0.95 / np.sqrt(L)

cost_fixed = []
err_fixed = []
iml12_fixed = \
    pyproximal.optimization.primaldual.PrimalDual(l2, l1iso, Gop,
                                                  tau=tau, mu=mu, theta=1.,
                                                  x0=np.zeros_like(img.ravel()),
                                                  gfirst=False, niter=300, show=True,
                                                  callback=lambda x: callback(x, l2, l1iso,
                                                                              Gop, cost_fixed,
                                                                              img.ravel(),
                                                                              err_fixed))
iml12_fixed = iml12_fixed.reshape(img.shape)

###############################################################################
# We do the same with the adaptive algorithm

cost_ada = []
err_ada = []
iml12_ada, steps = \
    pyproximal.optimization.primaldual.AdaptivePrimalDual(l2, l1iso, Gop,
                                                          tau=tau, mu=mu,
                                                          x0=np.zeros_like(img.ravel()),
                                                          niter=45, show=True, tol=0.05,
                                                          callback=lambda x: callback(x, l2, l1iso,
                                                                                      Gop, cost_ada,
                                                                                      img.ravel(),
                                                                                      err_ada))
iml12_ada = iml12_ada.reshape(img.shape)

###############################################################################
# Let's now compare the final results as well as the convergence curves of the
# two algorithms. We can see how the adaptive Primal-Dual produces a better
# estimate of the clean image in a much smaller number of iterations

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
axs[0].imshow(img, cmap='gray', vmin=0, vmax=255)
axs[0].set_title('Original')
axs[0].axis('off')
axs[0].axis('tight')
axs[1].imshow(noise_img, cmap='gray', vmin=0, vmax=255)
axs[1].set_title('Noisy')
axs[1].axis('off')
axs[1].axis('tight')
axs[2].imshow(iml12_fixed, cmap='gray', vmin=0, vmax=255)
axs[2].set_title('PD')
axs[2].axis('off')
axs[2].axis('tight')
axs[3].imshow(iml12_ada, cmap='gray', vmin=0, vmax=255)
axs[3].set_title('Adaptive PD')
axs[3].axis('off')
axs[3].axis('tight')

fig, axs = plt.subplots(2, 1, figsize=(12, 7))
axs[0].plot(cost_fixed, 'k', label='Fixed step')
axs[0].plot(cost_ada, 'r', label='Adaptive step')
axs[0].legend()
axs[0].set_title('Functional')
axs[1].plot(err_fixed, 'k', label='Fixed step')
axs[1].plot(err_ada, 'r', label='Adaptive step')
axs[1].set_title('MSE')
axs[1].legend()
plt.tight_layout()

fig, axs = plt.subplots(3, 1, figsize=(12, 7))
axs[0].plot(steps[0], 'k')
axs[0].set_title(r'$\tau^k$')
axs[1].plot(steps[1], 'k')
axs[1].set_title(r'$\mu^k$')
axs[2].plot(steps[2], 'k')
axs[2].set_title(r'$\alpha^k$')
plt.tight_layout();


