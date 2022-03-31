import numpy as np
from pyproximal.ProxOperator import _check_tau


def moreau(prox, x, tau, tol=1e-5, raiseerror=True, verb=False):
    r"""Moreau Identity.

    The Moreau identity defines a relation between the vector :math:`\mathbf{u}`,
    its proximal operator and its dual proximal operator and can be used to
    estimate one of the two operators given the knowledge of the other.

    Parameters
    ----------
    prox : :obj:`pyprox.ProxOperator`
        Proximal operator
    x : :obj:`np.ndarray`
        Vector
    tau : :obj:`float`
        Positive scalar weight
    tol : :obj:`float`, optional
        Moreau identity tolerance
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when Moreau identity is not
        satisfied
    verb : :obj:`bool`, optional
        Verbosity

    Notes
    -----
    The Moreau decomposition (or identity) is defined as:

    .. math::

        \mathbf{x} = \prox_{\tau f} (\mathbf{x}) +
        \tau \prox_{\frac{1}{\tau} f^*} (\frac{\mathbf{x}}{\tau})

    This routine is used to evaluate if the prox and dualprox implementations
    of a ``pyprox.ProxOperator`` satisfy such identity.

    """
    # compute prox
    p = prox.prox(x, tau)

    # compute dualprox
    pdual = tau * prox.proxdual(x / tau, 1. / tau)

    if verb:
        print('x: ', x)
        print('p + pdual: ', p + pdual)
        print('error: ', x - (p + pdual))
    if np.allclose(x, p + pdual, atol=tol):
        return True
    else:
        if raiseerror:
            raise ValueError('Moreau identity not verified')
        return False
