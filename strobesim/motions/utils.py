#!/usr/bin/env python
"""
utils.py -- utilities for the strobesim.motions module

"""
# Numeric
import numpy as np, warnings

# Hankel transforms
try:
    from hankel import SymmetricFourierTransform
except Exception as exc:
    warnings.warn("cannot import package 'hankel'; some functionality may be missing")
    SymmetricFourierTransform = None

def create_hankel_trans(d=3):
    """
    Create a global instance of hankel.SymmetricFourierTransform, since
    its instantiation is costly.

    args
    ----
        d       :   int, the number of spatial dimensions

    """
    if d == 2:
        global HankelTrans2D
        HankelTrans2D = SymmetricFourierTransform(ndim=2, N=10000, h=0.005)
    elif d == 3:
        global HankelTrans3D
        HankelTrans3D = SymmetricFourierTransform(ndim=3, N=10000, h=0.005)

def pdf_from_cf_rad(func_cf, x, d=3, **kwargs):
    """
    Evaluate a radially symmetric PDF defined in 3D, given its 
    characteristic function.

    important
    ---------
        The output of this function is not necessarily normalized. If the user
            wants to normalize it on 3D real space - for instance, to get the 
            PDF for the radial displacement from the origin of a trajectory in 
            3D space - they should use radnorm() with d = 3.

    args
    ----
        func_cf     :   function with signature (1D ndarray, **kwargs), the
                        characteristic function
        x           :   1D ndarray, the real-space points at which to evaluate
                        the PDF
        d           :   int, 2 or 3. The number of spatial dimensions.
        kwargs      :   to *func_cf*

    returns
    -------
        1D ndarray of shape x.shape, the PDF

    """
    if not d in [2, 3]:
        raise RuntimeError("Only dimensions 2 and 3 supported")

    # Generate global instance of the Hankel transformer if it
    # does not already exist
    trans_name = "HankelTrans{}D".format(d)
    if not trans_name in globals():
        create_hankel_trans(d=d)
    transform = globals()[trans_name]

    # The *hankel* package uses a function argument
    F = lambda j: func_cf(j, **kwargs)

    # Run the transform
    return transform.transform(F, x, ret_err=False, inverse=True)

def radnorm(r, pdf, d=2):
    """
    Given a PDF with radial symmetry, return the PDF for the radial distance
    from the origin. This is equivalent to taking the PDF, expressed in 
    hyperspherical coordinates, and marginalizing on all angular components.

    For instance, in 2D, we would do

        pdf_rad(r) = 2 pi int_{0}^{infty} r pdf(r) dr 

    Normalizing the Gaussian density with d = 2 would give a Rayleigh
    distribution, with d = 3 would give a Maxwell-Boltzmann distribution,
    and so on.

    This method approximates the PDF at a discrete set of radial points. For
    it to be accurate, the spacing of the support *r* must be small relative
    to local changes in the PDF.

    args
    ----
        r           :   1D ndarray of shape (n_r), the radial support
        pdf         :   1D ndarray of shape (n_r), the PDF as a function of 
                        the support
        d           :   int, the number of spatial dimensions

    returns
    -------
        1D ndarray of shape (n_r), the PDF for the radial distance from the
            origin

    """
    result = pdf * np.power(r, d-1)
    result /= result.sum()
    return result 
