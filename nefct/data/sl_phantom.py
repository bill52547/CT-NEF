# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: sl_phantom.py
@date: 8/21/2019
@desc:
'''
from nefct import nef_class
import numpy as np


def phantom(n = 256, p_type = 'Modified Shepp-Logan', ellipses = None):
    """
     phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
      "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
      if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
      the form
          [[I, a, b, x0, y0, phi],
           [I, a, b, x0, y0, phi],
           ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angles between the horizontal axis and
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """

    if (ellipses is None):
        ellipses = _select_phantom(p_type)
    elif (np.size(ellipses, 1) != 6):
        raise AssertionError("Wrong number of columns in user phantom")

    if '3d' not in p_type.lower():
        # Blank image
        p = np.zeros((n, n))

        # Create the pixel grid
        ygrid, xgrid = np.mgrid[-1:1:(1j * n), -1:1:(1j * n)]

        for ellip in ellipses:
            I = ellip[0]
            a2 = ellip[1] ** 2
            b2 = ellip[2] ** 2
            x0 = ellip[3]
            y0 = ellip[4]
            phi = ellip[5] * np.pi / 180  # Rotation angles in radians

            # Create the offset x and y values for the grid
            x = xgrid - x0
            y = ygrid - y0

            cos_p = np.cos(phi)
            sin_p = np.sin(phi)

            # Find the pixels within the ellipse
            locs = (((x * cos_p + y * sin_p) ** 2) / a2
                    + ((y * cos_p - x * sin_p) ** 2) / b2) <= 1

            # Add the ellipse intensity to those pixels
            p[locs] += I
    else:
        p = np.zeros((n, n, n))

        # rng = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
        rng = np.linspace(-1 + 2 / n, 1 - 2 / n, n)
        # Create the pixel grid
        xgrid, ygrid, zgrid = np.meshgrid(rng, rng, rng, indexing = 'ij')
        coord = np.vstack((xgrid.ravel(), ygrid.ravel(), zgrid.ravel()))
        for ellip in ellipses:
            A = ellip[0]
            asq = ellip[1] ** 2
            bsq = ellip[2] ** 2
            csq = ellip[3] ** 2
            x0 = ellip[4]
            y0 = ellip[5]
            z0 = ellip[6]
            phi = ellip[7] * np.pi / 180
            theta = ellip[8] * np.pi / 180
            psi = ellip[9] * np.pi / 180

            cphi = np.cos(phi)
            sphi = np.sin(phi)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)
            cpsi = np.cos(psi)
            spsi = np.sin(psi)

            alpha = np.array([[cpsi * cphi - ctheta * sphi * spsi, cpsi * sphi + ctheta * cphi *
                               spsi, spsi * stheta],
                              [-spsi * cphi - ctheta * sphi * cpsi,
                               -spsi * sphi + ctheta * cphi * cpsi,
                               cpsi * stheta],
                              [stheta * sphi, -stheta * cphi, ctheta]])

            x = xgrid * alpha[0, 0] + ygrid * alpha[1, 0] + zgrid * alpha[2, 0]
            y = xgrid * alpha[0, 1] + ygrid * alpha[1, 1] + zgrid * alpha[2, 1]
            z = xgrid * alpha[0, 2] + ygrid * alpha[1, 2] + zgrid * alpha[2, 2]

            idx = (x - x0) ** 2 / asq + (y - y0) ** 2 / bsq + (z - z0) ** 2 / csq <= 1
            p[idx] += A
    return p


def _select_phantom(name):
    if (name.lower() == 'shepp-logan'):
        e = _shepp_logan()
    elif (name.lower() == 'modified shepp-logan'):
        e = _mod_shepp_logan()
    elif (name.lower() == 'shepp-logan-3d'):
        e = _shepp_logan_3d()
    else:
        raise ValueError("Unknown phantom type: %s" % name)

    return e


def _shepp_logan():
    #  Standard head phantom, taken from Shepp & Logan
    return [[2, .69, .92, 0, 0, 0],
            [-.98, .6624, .8740, 0, -.0184, 0],
            [-.02, .1100, .3100, .22, 0, -18],
            [-.02, .1600, .4100, -.22, 0, 18],
            [.01, .2100, .2500, 0, .35, 0],
            [.01, .0460, .0460, 0, .1, 0],
            [.02, .0460, .0460, 0, -.1, 0],
            [.01, .0460, .0230, -.08, -.605, 0],
            [.01, .0230, .0230, 0, -.606, 0],
            [.01, .0230, .0460, .06, -.605, 0]]


def _shepp_logan_3d():
    #  Standard head phantom, taken from Shepp & Logan
    # return [[2, .69, .92, .81, 0, 0, 0, 0, 0, 0],
    #         [-.98, .6624, .8740, .78, 0, -.0184, 0, 0, 0, 0],
    #         [-.02, .1100, .3100, .22, .22, 0, 0, -18, 0, 10],
    #         [-.02, .1600, .4100, .28, -.22, 0, 0, 18, 0, 10],
    #         [.01, .2100, .2500, .41, 0, .35, -.15, 0, 0, 0],
    #         [.01, .0460, .0460, .05, 0, .1, .25, 0, 0, 0],
    #         [.02, .0460, .0460, .05, 0, -.1, .25, 0, 0, 0],
    #         [.01, .0460, .0230, .05, -.08, -.605, 0, 0, 0, 0],
    #         [.01, .0230, .0230, .02, 0, -.606, 0, 0, 0, 0],
    #         [.01, .0230, .0460, .02, .06, -.605, 0, 0, 0, 0]]

    return [[1, .6900, .920, .810, 0, 0, 0, 0, 0, 0],
            [- .8, .6624, .874, .780, 0, - .0184, 0, 0, 0, 0],
            [- .2, .1100, .310, .220, .22, 0, 0, - 18, 0, 10],
            [- .2, .1600, .410, .280, - .22, 0, 0, 18, 0, 10],
            [.1, .2100, .250, .410, 0, .35, - .15, 0, 0, 0],
            [.1, .0460, .046, .050, 0, .1, .25, 0, 0, 0],
            [.1, .0460, .046, .050, 0, - .1, .25, 0, 0, 0],
            [.1, .0460, .023, .050, - .08, - .605, 0, 0, 0, 0],
            [.1, .0230, .023, .020, 0, - .606, 0, 0, 0, 0],
            [.1, .0230, .046, .020, .06, - .605, 0, 0, 0, 0]]


def _mod_shepp_logan():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [[1, .69, .92, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18],
            [-.20, .1600, .4100, -.22, 0, 18],
            [.10, .2100, .2500, 0, .35, 0],
            [.10, .0460, .0460, 0, .1, 0],
            [.10, .0460, .0460, 0, -.1, 0],
            [.10, .0460, .0230, -.08, -.605, 0],
            [.10, .0230, .0230, 0, -.606, 0],
            [.10, .0230, .0460, .06, -.605, 0]]
