r"""
SV_DECONV
=====
This example implements the saptially varying PSF deconvolution algorithm
based on: Flicker & Rigaut, 2005 https://doi.org/10.1364/JOSAA.22.000504
"""

import numpy as np
from pylops import LinearOperator
from scipy.signal import fftconvolve
from pylops.basicoperators import FunctionOperator
from pylops.utils import dottest
from functools import partial

###############################################################################
# Sample data
im = np.random.randn(2748, 3840)
W = np.random.randn(20, np.prod(im.shape))
U = np.random.randn(np.prod(im.shape), 20)

# Setup Custom Linear Operator with Forward and Adjoint Models
# A) Using Function Operator
def forward(im, U, W, im_shape):
    im = im.reshape(*im_shape)
    for i in range(15):
        weight = W[i,:].reshape(*im_shape)
        psf_mode = U[:,i].reshape(*im_shape)
        im += fftconvolve(im* weight, psf_mode)
    return im.ravel()

def forward_c(im, U, W, im_shape):
    im = im.reshape(*im_shape)
    for i in range(15):
        weight = W[i,:].reshape(*im_shape)
        psf_mode = U[:,i].reshape(*im_shape)
        im += fftconvolve(im, np.flipud(np.fliplr(psf_mode)))* weight #psf_mode[::-1, ::-1]
    return im.ravel()

A = FunctionOperator(partial(forward, U=U, W=W, im_shape=im.shape), partial(forward_c, U, W, im_shape=im.shape), im.shape[0], im.shape[1])
print(dottest(A, im.shape[0], im.shape[1]))



# # Using Linear Opeartor Class
# class Blur(LinearOperator):
#     def __init___(self, shape):
#         super().__init__(shape=shape)

#     def _matvec(self, im):
#         im = im.reshape(self.shape)
#         for i in range(15):
#             weight = W[i,:].reshape(*self.shape)
#             psf_mode = U[:,i].reshape(*self.shape)
#             im += fftconvolve(im* weight, psf_mode)
#         return im.ravel()

#     def _rmatvec(self, im):
#         im = im.reshape(self.shape)
#         for i in range(15):
#             weight = W[i,:].reshape(*self.shape)
#             psf_mode = U[:,i].reshape(*self.shape)
#             im += fftconvolve(im, psf_mode[::-1, ::-1])* weight 
#         return im.ravel()

# Op = Blur(shape=im.shape, dtype=float)
# print(dottest(Op, nr=im.shape[0], nc=im.shape[1])) 
