r"""
Deblurring with multiple wavelet priors
=======================================
This tutorial considers the problem of deblurring an image that has been
convolved with a 2D Gaussian kernel and contaminated by random noise:

.. math::
    \mathbf{y} = \mathbf{B} \mathbf{x} + \mathbf{n}

where :math:`\mathbf{B}` is a 2D convolution operator and
:math:`\mathbf{n} \sim \mathcal{N}(0, \sigma^2)`.

Natural images are rarely well represented by a single sparsifying transform:
piecewise-constant regions with sharp edges are compactly described by the
Haar wavelet, whilst smoother, textured regions are better captured by
higher-order wavelets such as the Daubechies-4 (db4). We will therefore
promote sparsity in both domains at the same time by solving:

.. math::
    \argmin_\mathbf{x} \frac{1}{2}\|\mathbf{B}\mathbf{x}-\mathbf{y}\|_2^2 +
    \lambda_1 \|\mathbf{W}_1 \mathbf{x}\|_1 +
    \lambda_2 \|\mathbf{W}_2 \mathbf{x}\|_1

where :math:`\mathbf{W}_1` and :math:`\mathbf{W}_2` are the Haar and db4
2D wavelet transforms, respectively.

By stacking the two wavelet transforms into a single operator
:math:`\mathbf{K} = [\mathbf{W}_1^T, \mathbf{W}_2^T]^T`, the regularization
term becomes a separable function of :math:`\mathbf{K}\mathbf{x}`, whose
proximal operator is simply the stack of the proximal operators of the two
L1 norms. This problem can therefore be solved with the
:func:`pyproximal.optimization.primaldual.PrimalDual` solver
(Chambolle-Pock algorithm).

"""

import matplotlib.pyplot as plt
import numpy as np
import pylops
from pylops.utils.metrics import psnr

import pyproximal

plt.close("all")
np.random.seed(0)

###############################################################################
# Let's start by loading an image, converting it to grayscale, and cropping
# it to a square of size power of two (so that the wavelet transforms do not
# require any padding).
im = plt.imread("../testdata/butterfly.png")
im = np.mean(im[..., :3], axis=-1)

n = 256
h, w = im.shape
im = im[(h - n) // 2 : (h - n) // 2 + n, (w - n) // 2 : (w - n) // 2 + n]
ny, nx = im.shape

###############################################################################
# We now create a Gaussian blurring kernel and the associated
# :class:`pylops.signalprocessing.Convolve2D` operator. This is used to
# create the blurred data, to which we add white Gaussian noise with a
# signal-to-noise ratio of 30 dB.
nh = 11
sigh = 2.0
t = np.arange(-(nh // 2), nh // 2 + 1)
hy, hx = np.meshgrid(t, t, indexing="ij")
h = np.exp(-(hx**2 + hy**2) / (2 * sigh**2))
h /= h.sum()

Bop = pylops.signalprocessing.Convolve2D(
    dims=(ny, nx), h=h, offset=(nh // 2, nh // 2), dtype="float64"
)

yclean = Bop @ im.ravel()
snr_db = 30.0
sigman = np.linalg.norm(yclean) / np.sqrt(yclean.size) * 10 ** (-snr_db / 20)
y = yclean + sigman * np.random.randn(yclean.size)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(im, cmap="gray", vmin=0, vmax=1)
axs[0].set_title("True image")
axs[0].axis("off")
axs[1].imshow(y.reshape(ny, nx), cmap="gray", vmin=0, vmax=1)
axs[1].set_title(f"Blurred+noisy (PSNR={psnr(im, y.reshape(ny, nx), 1.0):.1f} dB)")
axs[1].axis("off")
axs[2].imshow(h, cmap="viridis")
axs[2].set_title(rf"Blurring kernel ($\sigma$={sigh})")
axs[2].axis("off")
plt.tight_layout()

###############################################################################
# Next, we define the two wavelet transforms and stack them vertically into
# the operator :math:`\mathbf{K}`.
W1op = pylops.signalprocessing.DWT2D(dims=(ny, nx), wavelet="haar", level=3)
W2op = pylops.signalprocessing.DWT2D(dims=(ny, nx), wavelet="db4", level=3)
Kop = pylops.VStack([W1op, W2op])
n1, n2 = W1op.shape[0], W2op.shape[0]

###############################################################################
# We are now ready to define the proximal operators. The data term is a
# :class:`pyproximal.L2` norm with the blurring operator, whilst the
# regularization term is a :class:`pyproximal.VStack` of two
# :class:`pyproximal.L1` norms, each acting on the portion of the vector
# :math:`\mathbf{K}\mathbf{x}` associated with one of the two wavelet
# transforms.
lam1 = 1e-3
lam2 = 1e-3

f = pyproximal.L2(Op=Bop, b=y)
g = pyproximal.VStack(
    [pyproximal.L1(sigma=lam1), pyproximal.L1(sigma=lam2)], nn=[n1, n2]
)

###############################################################################
# The Primal-Dual algorithm converges provided that
# :math:`\tau \mu \|\mathbf{K}\|_2^2 < 1`. Since each wavelet transform is
# (close to) orthogonal, we have :math:`\|\mathbf{K}\|_2^2 \le 2`, and choose
# the step lengths accordingly (with some safety margin). To appreciate the
# benefit of combining the two priors, we also solve the problem using a
# single wavelet transform at a time (for which
# :math:`\|\mathbf{W}_i\|_2^2 \le 1`).
tau = 0.4
mu = 1.0 / (tau * 2.5)
mu1 = 1.0 / (tau * 1.25)
niter = 400
x0 = np.zeros(ny * nx)

# Haar only
xhaar = pyproximal.optimization.primaldual.PrimalDual(
    f, pyproximal.L1(sigma=lam1), W1op, tau=tau, mu=mu1, x0=x0, niter=niter
)
xhaar = xhaar.reshape(ny, nx)

# db4 only
xdb4 = pyproximal.optimization.primaldual.PrimalDual(
    f, pyproximal.L1(sigma=lam2), W2op, tau=tau, mu=mu1, x0=x0, niter=niter
)
xdb4 = xdb4.reshape(ny, nx)

# Haar + db4
xdual = pyproximal.optimization.primaldual.PrimalDual(
    f, g, Kop, tau=tau, mu=mu, x0=x0, niter=niter, show=True
)
xdual = xdual.reshape(ny, nx)

###############################################################################
# Finally, we compare the three reconstructions.
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
for ax, img, title in zip(
    axs.ravel(),
    [y.reshape(ny, nx), xhaar, xdb4, xdual],
    ["Blurred+noisy", "Haar prior", "db4 prior", "Haar + db4 priors"],
    strict=True,
):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"{title} (PSNR={psnr(im, img, 1.0):.1f} dB)")
    ax.axis("off")
plt.tight_layout()

###############################################################################
# Both single-prior reconstructions already remove most of the blur. The
# Haar prior favours piecewise-constant solutions and is therefore prone
# to blocky artefacts, whilst the db4 prior produces smoother images. By
# combining the two priors, we obtain a reconstruction that benefits from
# the strengths of both transforms.
