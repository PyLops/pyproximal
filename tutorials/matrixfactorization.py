r"""
Low-Rank completion via Matrix factorization
============================================
In this tutorial we will present another example of low-rank matrix completion.
This time, however, we will not leverage SVD to find a low-rank representation
of the matrix, instead we will look for two matrices whose inner product can
represent the matrix we are after.

More specifically we will consider the following forward problem:

.. math::
    \mathbf{X},\mathbf{Y} = \argmin_{\mathbf{X}, \mathbf{Y}} \frac{1}{2}
    \|\mathbf{XY}-\mathbf{A}\|_F^2 + \delta_{\mathbf{X}\ge0} + \delta_{\mathbf{Y}\ge0}

where the non-negativity constraint (:math:`\delta_{\cdot \ge0}`) is simply
implemented using a `Box` proximal operator.

"""
import numpy as np
import matplotlib.pyplot as plt
import pylops
import pyproximal

from scipy import misc

plt.close('all')
np.random.seed(10)


def callback(x, y, n, m, k, xtrue, snr_hist):
    snr_hist.append(pylops.utils.metrics.snr(xtrue, x.reshape(n, k) @ y.reshape(k, m)))


###############################################################################
# Let's start by creating the matrix we want to factorize
n, m, k = 100, 90, 10
X = np.maximum(np.random.normal(0, 1, (n, k)), 0) + 1.
Y = np.maximum(np.random.normal(0, 1, (k, m)), 0) + 1.

A = X @ Y

###############################################################################
# We can now define the Box operators and the Low-Rank factorized operator. To
# do so we need some initial guess of :math:`\mathbf{X}` and :math:`\mathbf{Y}`
# that we create using the same distribution of the original ones.

nn1 = pyproximal.Box(lower=0)
nn2 = pyproximal.Box(lower=0)

Xin = np.maximum(np.random.normal(0, 1, (n, k)), 0) + 1.
Yin = np.maximum(np.random.normal(0, 1, (k, m)), 0) + 1.
Hop = pyproximal.utils.bilinear.LowRankFactorizedMatrix(Xin, Yin, A.ravel())

###############################################################################
# We are now ready to run the PALM algorithm
snr_palm = []
Xpalm, Ypalm = \
    pyproximal.optimization.palm.PALM(Hop, nn1, nn2, Xin.ravel(), Yin.ravel(),
                                      gammaf=2, gammag=2, niter=2000, show=True,
                                      callback=lambda x, y: callback(x, y, n, m, k,
                                                                     A, snr_palm))
Xpalm, Ypalm = Xpalm.reshape(Xin.shape), Ypalm.reshape(Yin.shape)
Apalm = Xpalm @ Ypalm

fig, axs = plt.subplots(1, 5, figsize=(14, 3))
fig.suptitle('PALM')
axs[0].imshow(Xpalm, cmap='gray')
axs[0].set_title('Xest')
axs[0].axis('tight')
axs[1].imshow(Ypalm, cmap='gray')
axs[1].set_title('Yest')
axs[1].axis('tight')
axs[2].imshow(A, cmap='gray', vmin=10, vmax=37)
axs[2].set_title('True')
axs[2].axis('tight')
axs[3].imshow(Apalm, cmap='gray', vmin=10, vmax=37)
axs[3].set_title('Reconstructed')
axs[3].axis('tight')
axs[4].imshow(A - Apalm, cmap='gray', vmin=-.1, vmax=.1)
axs[4].set_title('Reconstruction error')
axs[4].axis('tight')
fig.tight_layout()

###############################################################################
# Similarly we run the PALM algorithm with backtracking
snr_palmbt = []
Xpalmbt, Ypalmbt = \
    pyproximal.optimization.palm.PALM(Hop, nn1, nn2, Xin.ravel(), Yin.ravel(),
                                      gammaf=None, gammag=None, niter=2000, show=True,
                                      callback=lambda x, y: callback(x, y, n, m, k,
                                                                     A, snr_palmbt))
Xpalmbt, Ypalmbt = Xpalmbt.reshape(Xin.shape), Ypalmbt.reshape(Yin.shape)
Apalmbt = Xpalmbt @ Ypalmbt

fig, axs = plt.subplots(1, 5, figsize=(14, 3))
fig.suptitle('PALM with back-tracking')
axs[0].imshow(Xpalmbt, cmap='gray')
axs[0].set_title('Xest')
axs[0].axis('tight')
axs[1].imshow(Ypalmbt, cmap='gray')
axs[1].set_title('Yest')
axs[1].axis('tight')
axs[2].imshow(A, cmap='gray', vmin=10, vmax=37)
axs[2].set_title('True')
axs[2].axis('tight')
axs[3].imshow(Apalmbt, cmap='gray', vmin=10, vmax=37)
axs[3].set_title('Reconstructed')
axs[3].axis('tight')
axs[4].imshow(A - Apalmbt, cmap='gray', vmin=-.1, vmax=.1)
axs[4].set_title('Reconstruction error')
axs[4].axis('tight')
fig.tight_layout()

###############################################################################
# And the iPALM algorithm
snr_ipalm = []
Xipalm, Yipalm = \
    pyproximal.optimization.palm.iPALM(Hop, nn1, nn2, Xin.ravel(), Yin.ravel(),
                                       gammaf=2, gammag=2, a=[0.8, 0.8],
                                       niter=2000, show=True,
                                       callback=lambda x, y: callback(x, y, n, m, k,
                                                                      A, snr_ipalm))
Xipalm, Yipalm = Xipalm.reshape(Xin.shape), Yipalm.reshape(Yin.shape)
Aipalm = Xipalm @ Yipalm

fig, axs = plt.subplots(1, 5, figsize=(14, 3))
fig.suptitle('iPALM')
axs[0].imshow(Xipalm, cmap='gray')
axs[0].set_title('Xest')
axs[0].axis('tight')
axs[1].imshow(Yipalm, cmap='gray')
axs[1].set_title('Yest')
axs[1].axis('tight')
axs[2].imshow(A, cmap='gray', vmin=10, vmax=37)
axs[2].set_title('True')
axs[2].axis('tight')
axs[3].imshow(Aipalm, cmap='gray', vmin=10, vmax=37)
axs[3].set_title('Reconstructed')
axs[3].axis('tight')
axs[4].imshow(A - Aipalm, cmap='gray', vmin=-.1, vmax=.1)
axs[4].set_title('Reconstruction error')
axs[4].axis('tight')
fig.tight_layout()

###############################################################################
# And finally compare the converge behaviour of the three methods
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(snr_palm, 'k', lw=2, label='PALM')
ax.plot(snr_palmbt, 'r', lw=2, label='PALM')
ax.plot(snr_ipalm, 'g', lw=2, label='iPALM')
ax.grid()
ax.legend()
ax.set_title('SNR')
ax.set_xlabel('# Iteration')
fig.tight_layout()
