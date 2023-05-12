r"""
MRI Imaging and Segmentation of Brain
=====================================
This tutorial considers the well-known problem of MRI imaging, where given the availability of a sparsely sampled
KK-spectrum, one is tasked to reconstruct the underline spatial luminosity of an object under observation. In
this specific case, we will be using an example from `Corona et al., 2019, Enhancing joint reconstruction and
segmentation with non-convex Bregman iteration`.

We first consider the imaging problem defined by the following cost functuon

.. math::
    \argmin_\mathbf{x} \|\mathbf{y}-\mathbf{Ax}\|_2^2 + \alpha TV(\mathbf{x})

where the operator :math:`\mathbf{A}` performs a 2D-Fourier transform followed by sampling of the KK plane, :math:`\mathbf{x}`
is the object of interest and :math:`\mathbf{y}` the set of available Fourier coefficients.

Once the model is reconstructed, we solve a second inverse problem with the aim of segmenting the retrieved object into
:math:`N` classes of different luminosity.
"""
import numpy as np
import matplotlib.pyplot as plt
import pylops
from scipy.io import loadmat

import pyproximal

plt.close('all')
np.random.seed(10)

###############################################################################
# Let's start by loading the data and the sampling mask
mat = loadmat('../testdata/brainphantom.mat')
mat1 = loadmat('../testdata/spiralsampling.mat')
gt = mat['gt']
seggt = mat['gt_seg']
sampling = mat1['samp']
sampling1 = np.fft.ifftshift(sampling)

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].imshow(gt, cmap='gray')
axs[0].axis('tight')
axs[0].set_title("Object")
axs[1].imshow(seggt, cmap='Accent')
axs[1].axis('tight')
axs[1].set_title("Segmentation")
axs[2].imshow(sampling, cmap='gray')
axs[2].axis('tight')
axs[2].set_title("Sampling mask")
plt.tight_layout()

###############################################################################
# We can now create the MRI operator
Fop = pylops.signalprocessing.FFT2D(dims=gt.shape)
Rop = pylops.Restriction(gt.size, np.where(sampling1.ravel() == 1)[0],
                         dtype=np.complex128)
Dop = Rop * Fop

# KK spectrum
GT = Fop * gt.ravel()
GT = GT.reshape(gt.shape)

# Data (Masked KK spectrum)
d = Dop * gt.ravel()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(np.fft.fftshift(np.abs(GT)), vmin=0, vmax=1, cmap='gray')
axs[0].axis('tight')
axs[0].set_title("Spectrum")
axs[1].plot(np.fft.fftshift(np.abs(d)), 'k', lw=2)
axs[1].axis('tight')
axs[1].set_title("Masked Spectrum")
plt.tight_layout()

###############################################################################
# Let's try now to reconstruct the object from its measurement. The simplest
# approach entails simply filling the missing values in the KK spectrum with
# zeros and applying inverse FFT.

GTzero = sampling1 * GT
gtzero = (Fop.H * GTzero).real

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(gt, cmap='gray')
axs[0].axis('tight')
axs[0].set_title("True Object")
axs[1].imshow(gtzero, cmap='gray')
axs[1].axis('tight')
axs[1].set_title("Zero-filling Object")
plt.tight_layout()

###############################################################################
# We can now do better if we introduce some prior information in the form of
# TV on the solution

with pylops.disabled_ndarray_multiplication():
    sigma = 0.04
    l1 = pyproximal.proximal.L21(ndim=2)
    l2 = pyproximal.proximal.L2(Op=Dop, b=d.ravel(), niter=50, warm=True)
    Gop = sigma * pylops.Gradient(dims=gt.shape, edge=True, kind='forward', dtype=np.complex128)

    L = sigma ** 2 * 8
    tau = .99 / np.sqrt(L)
    mu = .99 / np.sqrt(L)

    gtpd = pyproximal.optimization.primaldual.PrimalDual(l2, l1, Gop, x0=np.zeros(gt.size),
                                                         tau=tau, mu=mu, theta=1.,
                                                         niter=100, show=True)
    gtpd = np.real(gtpd.reshape(gt.shape))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(gt, cmap='gray')
axs[0].axis('tight')
axs[0].set_title("True Object")
axs[1].imshow(gtpd, cmap='gray')
axs[1].axis('tight')
axs[1].set_title("TV-reg Object")
plt.tight_layout()

###############################################################################
# Finally we segment our reconstructed model into 4 classes.

cl = np.array([0.01, 0.43, 0.65, 0.8])
ncl = len(cl)
segpd_prob, segpd = \
    pyproximal.optimization.segmentation.Segment(gtpd, cl, 1., 0.001,
                                                 niter=10, show=True,
                                                 kwargs_simplex=dict(engine='numba'))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(seggt, cmap='Accent')
axs[0].axis('tight')
axs[0].set_title("True Classes")
axs[1].imshow(segpd, cmap='Accent')
axs[1].axis('tight')
axs[1].set_title("Estimated Classes")
plt.tight_layout()

fig, axs = plt.subplots(1, 4, figsize=(15, 6))
for i, ax in enumerate(axs):
    ax.imshow(segpd_prob[:, i].reshape(gt.shape), cmap='Reds')
    axs[i].axis('tight')
    axs[i].set_title(f"Class {i}")
plt.tight_layout()