r"""
Regularization by Denoising (RED)
=================================
This is a follow up tutorial to the :ref:`sphx_glr_tutorials_plugandplay.py` tutorial,
showcasing an competitive technical of the famous Plug-and-Play method called
Regularization by Denoising (RED).

The Plug-and-Play algorithm leverges a user-defined denoiser in place of the proximal
operator of the regularization term in the solution of an inverse problem, ultimately
acting as an implicit prior; RED, instead, defines an the following
explicit regularization term

.. math::
    RED(\mathbf{x}) = \sigma\mathbf{x}^T (\mathbf{x} - f_{\sigma_d}(\mathbf{x}))

where the dot-product of the sought after model and residual from the action of
the denoiser is minimized.

Let's consider again a simplified MRI experiment, where the
data is created by appling a 2D Fourier Transform to the input model and
by randomly sampling 60% of its values, and the
`BM3D <https://pypi.org/project/bm3d>`_ method as the denoiser of choice.

Two different solvers will be compared, namely:

- Gradient descent, which simply uses the gradient of the data misfit term and that
  of the (now well defined and differentiable) regularization term;
- ADMM, where the proximal of RED is solved using a fixed-point iteration.
- Fixed-point method.

"""

import bm3d
import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.config import set_ndarray_multiplication
from pylops.utils.metrics import snr

import pyproximal

plt.close("all")
np.random.seed(0)
set_ndarray_multiplication(False)

###############################################################################
# We start by loading the famous Shepp logan phantom and creating the
# modelling operator
x = np.load("../testdata/shepp_logan_phantom.npy")
x = x / x.max()
ny, nx = x.shape

perc_subsampling = 0.6
nxsub = int(np.round(ny * nx * perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(ny * nx))[:nxsub])
Rop = pylops.Restriction(ny * nx, iava, dtype=np.complex128)
Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))
Op = Rop * Fop

###############################################################################
# We now create and display the data alongside the model
y = Rop * Fop * x.ravel()
yfft = Fop * x.ravel()
yfft = np.fft.fftshift(yfft.reshape(ny, nx))

ymask = Rop.mask(Fop * x.ravel())
ymask = ymask.reshape(ny, nx)
ymask.data[:] = np.fft.fftshift(ymask.data)
ymask.mask[:] = np.fft.fftshift(ymask.mask)

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(x, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(np.abs(yfft), vmin=0, vmax=1, cmap="rainbow")
axs[1].set_title("Full data")
axs[1].axis("tight")
axs[2].imshow(np.abs(ymask), vmin=0, vmax=1, cmap="rainbow")
axs[2].set_title("Sampled data")
axs[2].axis("tight")
plt.tight_layout()

###############################################################################
# At this point we create a denoiser instance using the BM3D algorithm and use
# the gradient descent solver that we wrote at the start


def callback(x, xtrue, errhist):
    errhist.append(np.linalg.norm(x - xtrue))


def sigmad(iiter):
    return 0.1 * 0.99**iiter


# BM3D denoiser
denoiser = lambda x, sigma: bm3d.bm3d(
    np.real(x), sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
)

l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel())
red = pyproximal.proximal.RED(denoiser, x.shape, sigma=0.4, sigmad=sigmad, call=False)

errhistgd = []
xredgd = pyproximal.optimization.red.RED(
    l2,
    red,
    x0=np.zeros(x.size),
    solver="gradientdescent",
    alpha=0.5,
    niter=50,
    callback=lambda xx: callback(xx, x.ravel(), errhistgd),
    show=True,
)
xredgd = np.real(xredgd.reshape(x.shape))

################################################################################
# And now we use the ADMM solver

L = np.real((Op.H * Op).eigs(neigs=1, which="LM")[0])
tau = 1.0 / L

# BM3D denoiser
denoiser = lambda x, sigma: bm3d.bm3d(
    np.real(x), sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
)

# ADMM-RED
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=10, warm=True)
red = pyproximal.proximal.RED(
    denoiser, x.shape, sigma=0.4, sigmad=sigmad, niter=5, warm=True, call=False
)

errhistadmm = []
xredadmm = pyproximal.optimization.red.RED(
    l2,
    red,
    x0=np.zeros(x.size),
    solver=pyproximal.optimization.primal.ADMM,
    tau=tau,
    niter=50,
    show=True,
    callback=lambda xx: callback(xx, x.ravel(), errhistadmm),
)[0]
xredadmm = np.real(xredadmm.reshape(x.shape))

###############################################################################
# And finally we use the Fixed-Point solver

# BM3D
xshape = x.shape
denoiser = lambda x, sigma: bm3d.bm3d(
    x.real.reshape(xshape), sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
).ravel()

# FP-RED
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel())
red = pyproximal.proximal.RED(
    denoiser, x.shape, sigma=0.4, sigmad=sigmad, niter=5, warm=True, call=False
)

errhistfp = []
xredfp = pyproximal.optimization.red.RED(
    l2,
    red,
    x0=np.zeros(x.size),
    solver="fixedpoint",
    niter=50,
    niter_inner=10,
    callback=lambda xx: callback(xx, x.ravel(), errhistfp),
    show=True,
)
xredfp = np.real(xredfp.reshape(x.shape))

###############################################################################
# Let's finally compare the results and the error convergence of the three
# variations of RED

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
axs[0].imshow(x, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(xredgd, vmin=0, vmax=1, cmap="gray")
axs[1].set_title(f"GD-RED (SNR={snr(x, xredgd):.2f} dB)")
axs[1].axis("tight")
axs[2].imshow(xredadmm, vmin=0, vmax=1, cmap="gray")
axs[2].set_title(f"ADMM-RED (SNR={snr(x, xredadmm):.2f} dB)")
axs[2].axis("tight")
axs[3].imshow(xredfp, vmin=0, vmax=1, cmap="gray")
axs[3].set_title(f"FP-RED (SNR={snr(x, xredfp):.2f} dB)")
axs[3].axis("tight")
plt.tight_layout()

plt.figure(figsize=(12, 3))
plt.semilogy(errhistgd, "k", lw=2, label="GD")
plt.semilogy(errhistadmm, "r", lw=2, label="ADMM")
plt.semilogy(errhistfp, "b", lw=2, label="FP")

plt.title("Error norm")
plt.legend()
plt.tight_layout()
