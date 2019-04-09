# -*- coding: utf-8 -*-
"""
Basic tomography toolbox.	
"""
import numpy as np
from scipy.ndimage.interpolation import rotate


def apply_mask(array):
    """
    Apply round mask with radius N on square array with dimensions (N,N).
    """

    N = array.shape[0]
    x,y = np.ogrid[-1:1:1j*N, -1:1:1j*N]
    mask = (x*x+y*y <= 1.)
    return array*mask

def forwardproject(tomo, theta):
    """
    Project the given tomographic slice in direction theta.
    It is assumed that the provided tomographic slice is (1) square
    and (2) zero outside the disc inscribed in this square.
    """
    # Rotate the image
    tomo_rot = rotate(tomo, -theta, reshape=False, order=1)
    
    # Collapse along first axis
    proj = np.sum(tomo_rot, axis=0)

    return proj

def backproject(proj, angle):
    """
    Backproject the given projection onto the tomographic slice.
    """

    N = len(proj)

    proj_spread = np.tile(proj, (N,1))

    rot_proj = rotate(proj_spread, angle, reshape=False, order=1)

    rot_masked = apply_mask(rot_proj)

    return rot_masked

def fbp(sinogram, filter_type='ramp'):
    """
    Perform FBP reconstruction from a given sinogram.

    Parameters
    ----------
    sinogram: ndarray
        The input sinogram, with dimensions (N, Nangles)
    """

    # Initialize
    N,Nangles = sinogram.shape
    angle_list = 360. * np.arange(Nangles)/Nangles

    # Generate basic ramp filter
    q = np.fft.fftfreq(N).reshape((N,1)) # Fourier coordinates
    f = 2.*abs(q) # ramp filter

    filter_type = filter_type.lower().strip() # have filter type all lower case and without leading or trailing whitespace

    if filter_type == "ramp":
        pass # do nothing if ramp filter is set
    elif filter_type == "hamming":
        f = f * (0.54 + 0.46 * np.cos(2*np.pi*q))

    # Filter sinogram in Fourier space in detector direction
    sino_ft = np.fft.fft(sinogram, axis=0)
    sino_ft *= f
    sino_filtered = np.real(np.fft.ifft(sino_ft, axis=0))

    reconstructed_image = np.zeros((N,N))

    for i, angle in enumerate(angle_list):
        reconstructed_image += backproject(sino_filtered[:,i], angle)
    return reconstructed_image/Nangles






