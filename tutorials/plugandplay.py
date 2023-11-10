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
import numpy as np
import matplotlib.pyplot as plt
import pylops

import pyproximal
import bm3d

from pylops.config import set_ndarray_multiplication

plt.close('all')
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
# as Plug-and-Play Prior to the PG and ADMM algorithms

def callback(x, xtrue, errhist):
    errhist.append(np.linalg.norm(x - xtrue))

Op = Rop * Fop
L = np.real((Op.H*Op).eigs(neigs=1, which='LM')[0])
tau = 1./L
sigma = 0.05

l2 = pyproximal.proximal.L2(Op=Op, b=y.ravel(), niter=50, warm=True)

# BM3D denoiser
denoiser = lambda x, tau: bm3d.bm3d(np.real(x), sigma_psd=sigma * tau,
                                    stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

# PG-Pnp
errhistpg = []
xpnppg = pyproximal.optimization.pnp.PlugAndPlay(l2, denoiser, x.shape,
                                                 solver=pyproximal.optimization.primal.ProximalGradient,
                                                 tau=tau, x0=np.zeros(x.size),
                                                 niter=40,
                                                 acceleration='fista',
                                                 show=True,
                                                 callback=lambda xx: callback(xx, x.ravel(),
                                                                              errhistpg))
xpnppg = np.real(xpnppg.reshape(x.shape))

# ADMM-PnP
errhistadmm = []
xpnpadmm = pyproximal.optimization.pnp.PlugAndPlay(l2, denoiser, x.shape,
                                                   solver=pyproximal.optimization.primal.ADMM,
                                                   tau=tau, x0=np.zeros(x.size),
                                                   niter=40, show=True,
                                                   callback=lambda xx: callback(xx, x.ravel(),
                                                                                errhistadmm))[0]
xpnpadmm = np.real(xpnpadmm.reshape(x.shape))

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
axs[0].imshow(x, vmin=0, vmax=1, cmap="gray")
axs[0].set_title("Model")
axs[0].axis("tight")
axs[1].imshow(xpnppg, vmin=0, vmax=1, cmap="gray")
axs[1].set_title("PG-PnP Inversion")
axs[1].axis("tight")
axs[2].imshow(xpnpadmm, vmin=0, vmax=1, cmap="gray")
axs[2].set_title("ADMM-PnP Inversion")
axs[2].axis("tight")
plt.tight_layout()

###############################################################################
# Finally, let's compare the error convergence of the two variations of PnP

plt.figure(figsize=(12, 3))
plt.plot(errhistpg, 'k', lw=2, label='PG')
plt.plot(errhistadmm, 'r', lw=2, label='ADMM')
plt.title("Error norm")
plt.legend()
plt.tight_layout()
