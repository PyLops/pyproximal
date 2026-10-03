r"""
Plug and Play Priors
====================
In this tutorial we will consider a rather atypical proximal algorithm.
In their seminal work, *Venkatakrishnan et al. [2021], Plug-and-Play Priors
for Model Based Reconstruction* showed that the y-update in the ADMM algorithm
can be interpreted as a denoising problem. The authors therefore suggested to
replace the regularizer of the original problem with any denoising algorithm
of choice (even if it does not have a known proximal). The proposed algorithm
has shown great performance in a variety of inverse problems.

As an example, we will consider a simplified MRI experiment, where the
data is created by appling a 2D Fourier Transform to the input model and
by randomly sampling 60% of its values. We will use the famous
`BM3D <https://pypi.org/project/bm3d>`_ as the denoiser, but any other denoiser
of choice can be used instead!

Finally, whilst in the original paper, PnP is associated to the ADMM solver, subsequent
research showed that the same principle can be applied to pretty much any proximal
solver. We will show how to pass a solver of choice to our
:func:`pyproximal.optimization.pnp.PlugAndPlay` solver.

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
# Let's start by loading the famous Shepp logan phantom and creating the
# modelling operator
x = np.load("../testdata/shepp_logan_phantom.npy")
x = x / x.max()
ny, nx = x.shape

perc_subsampling = 0.6
nxsub = int(np.round(ny * nx * perc_subsampling))
iava = np.sort(np.random.permutation(np.arange(ny * nx))[:nxsub])
Rop = pylops.Restriction(ny * nx, iava, dtype=np.complex128)
Fop = pylops.signalprocessing.FFT2D(dims=(ny, nx))

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
# as Plug-and-Play Prior to the ADMM, PG and HQS algorithms


def callback(x, xtrue, errhist):
    errhist.append(np.linalg.norm(x - xtrue))


Op = Rop * Fop
L = np.real((Op.H * Op).eigs(neigs=1, which="LM")[0])
tau = 1.0 / L
sigma = 0.05

# BM3D denoiser
denoiser = lambda x, tau: bm3d.bm3d(
    np.real(x), sigma_psd=sigma * tau, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING
)

# ADMM-PnP
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=50, warm=True)

errhistadmm = []
xpnpadmm = pyproximal.optimization.pnp.PlugAndPlay(
    l2,
    denoiser,
    x.shape,
    solver=pyproximal.optimization.primal.ADMM,
    tau=tau,
    x0=np.zeros(x.size, dtype=np.complex128),
    niter=40,
    show=True,
    callback=lambda xx: callback(xx, x.ravel(), errhistadmm),
)[0]
xpnpadmm = np.real(xpnpadmm.reshape(x.shape))

# PG-Pnp
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=50, warm=True)

errhistpg = []
xpnppg = pyproximal.optimization.pnp.PlugAndPlay(
    l2,
    denoiser,
    x.shape,
    solver=pyproximal.optimization.primal.ProximalGradient,
    tau=tau,
    x0=np.zeros(x.size, dtype=np.complex128),
    niter=40,
    acceleration="fista",
    show=True,
    callback=lambda xx: callback(xx, x.ravel(), errhistpg),
)
xpnppg = np.real(xpnppg.reshape(x.shape))

# HQS-PnP
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=50, warm=True)

tau_hqs = 1.0 / L * 0.99 ** (np.arange(40))
errhisthqs = []
xpnphqs = pyproximal.optimization.pnp.PlugAndPlay(
    l2,
    denoiser,
    x.shape,
    solver=pyproximal.optimization.primal.HQS,
    tau=tau_hqs,
    x0=np.zeros(x.size, dtype=np.complex128),
    niter=40,
    show=True,
    callback=lambda xx: callback(xx, x.ravel(), errhisthqs),
)[0]
xpnphqs = np.real(xpnphqs.reshape(x.shape))

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
axs[0].imshow(x, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(xpnpadmm, vmin=0, vmax=1, cmap="gray")
axs[1].set_title(f"ADMM-PnP (SNR={snr(x, xpnpadmm):.2f} dB)")
axs[1].axis("tight")
axs[2].imshow(xpnppg, vmin=0, vmax=1, cmap="gray")
axs[2].set_title(f"PG-PnP (SNR={snr(x, xpnppg):.2f} dB)")
axs[2].axis("tight")
axs[3].imshow(xpnphqs, vmin=0, vmax=1, cmap="gray")
axs[3].set_title(f"HQS-PnP (SNR={snr(x, xpnphqs):.2f} dB)")
axs[3].axis("tight")
plt.tight_layout()

###############################################################################
# Finally, the attentive reader may have noticed that in the HQS server a
# continuation strategy was used for the `tau` parameter; whilst this is
# strictly needed for HQS to converge, there is a consensus in the literature
# that also other solvers should benefit from adopting the same strategy
# when used with a PnP prior. This can be in fact interpreted as reducing
# the strength of the denoiser as iterations progress and the estimate comes
# closer to the true solution.
#
# While our :func:`pyproximal.optimization.primal.ADMM` solver does currently
# not offer relaxation out-of-the-box, this can be achieved pretty easily
# by creating an auxiliary `Denoiser` class with a `decay` parameter as
# shown below.


class Denoiser:
    def __init__(self, sigma, decay):
        self.sigma = sigma
        self.decay = decay
        self.iiter = 0

    def denoise(self, x, tau):
        xden = bm3d.bm3d(
            np.real(x),
            sigma_psd=self.decay[self.iiter] * self.sigma * tau,
            stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING,
        )
        self.iiter += 1
        return xden


# ADMM-PnP with relaxation
denoiser = Denoiser(sigma, decay=0.99 ** (np.arange(40)))
l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=50, warm=True)

errhistadmm1 = []
xpnpadmm1 = pyproximal.optimization.pnp.PlugAndPlay(
    l2,
    denoiser.denoise,
    x.shape,
    solver=pyproximal.optimization.primal.ADMM,
    tau=tau,
    x0=np.zeros(x.size, dtype=np.complex128),
    niter=40,
    show=True,
    callback=lambda xx: callback(xx, x.ravel(), errhistadmm1),
)[0]
xpnpadmm1 = np.real(xpnpadmm1.reshape(x.shape))

fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
axs[0].imshow(x, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(xpnpadmm, vmin=0, vmax=1, cmap="gray")
axs[1].set_title(f"ADMM-PnP (SNR={snr(x, xpnpadmm):.2f} dB)")
axs[1].axis("tight")
axs[2].imshow(xpnpadmm1, vmin=0, vmax=1, cmap="gray")
axs[2].set_title(f"ADMM-PnP with rel. (SNR={snr(x, xpnpadmm1):.2f} dB)")
axs[2].axis("tight")
plt.tight_layout()

###############################################################################
# Let's finally compare the error convergence of the four variations of PnP

plt.figure(figsize=(12, 3))
plt.semilogy(errhistadmm, "k", lw=2, label="ADMM")
plt.semilogy(errhistpg, "r", lw=2, label="PG")
plt.semilogy(errhisthqs, "b", lw=2, label="HQS")
plt.semilogy(errhistadmm1, "--b", lw=2, label="ADMM with rel.")
plt.title("Error norm")
plt.legend()
plt.tight_layout()

###############################################################################
# This final results clearly shows the importance of relaxation also for ADMM.
