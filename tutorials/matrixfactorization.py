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
Xest, Yest = \
    pyproximal.optimization.palm.PALM(Hop, nn1, nn2, Xin.ravel(), Yin.ravel(),
                                      gammaf=2, gammag=2, niter=2000, show=True)
Xest, Yest = Xest.reshape(Xin.shape), Yest.reshape(Yin.shape)
Aest = Xest @ Yest

###############################################################################
# And finally we display the individual components and the reconstructed matrix

fig, axs = plt.subplots(1, 5, figsize=(14, 3))
axs[0].imshow(Xest, cmap='gray')
axs[0].set_title('Xest')
axs[0].axis('tight')
axs[1].imshow(Yest, cmap='gray')
axs[1].set_title('Yest')
axs[1].axis('tight')
axs[2].imshow(A, cmap='gray')
axs[2].set_title('True')
axs[2].axis('tight')
axs[3].imshow(Aest, cmap='gray')
axs[3].set_title('Reconstructed')
axs[3].axis('tight')
axs[4].imshow(A-Aest, cmap='gray')
axs[4].set_title('Reconstruction error')
axs[4].axis('tight')
fig.tight_layout()
