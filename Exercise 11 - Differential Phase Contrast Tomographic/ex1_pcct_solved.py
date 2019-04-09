"""
27.07.2017
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Maximilian Teuffenbach)

ex1_pcct.py

This exercise will be about a very simple implementation of tomographic
reconstruction for differential phase contrast.

In this exercise you will compare two different reconstruction methods for
differential phase contrast data. 

The first method consists of two steps: First you perform a 1D integration 
on the differential phase projections to acquire non-differential phase 
projections. These projections are then subsequently reconstructed using 
filtered backprojection with a ramp filter.
The second method directly reconstructs the differential phase projections
using filtered backprojection with the complex Hilbert filter.

The dataset consists of a single line of a scan of a plastic tube containing
an ex-vivo baby mouse fixated in formalin. The whole tube was submerged in a
water bath (which was also present while acquiring the reference scan).

You need to replace the ??? in the code with the required commands.
"""

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def apply_mask(array):
    """
    Apply round mask with radius N on square array with dimensions (N,N).
    """

    N = array.shape[0]
    x, y = np.ogrid[-1:1:1j*N, -1:1:1j*N]
    mask = (x*x + y*y <= 1.)
    return array*mask


def fbp(sinogram, filter_type='ramp'):
    """
    Perform FBP reconstruction from a given sinogram.

    Parameters
    ----------
    sinogram: ndarray
        The input sinogram, with dimensions (N, Nangles)
    filter_type : string
        The filter kernel that should be used.
        Default is "ramp". For differential data use "hilbert".
    """

    # Initialize
    Nangles, N = sinogram.shape
    angle_list = 360. * np.arange(Nangles) / Nangles

    # Generate basic ramp filter
    q = np.fft.fftfreq(N)  # Fourier coordinates

    # have filter type all lower case & without leading or trailing whitespace
    filter_type = filter_type.lower().strip() 

    if filter_type == 'ramp':
        f = np.abs(q)  # ramp filter
    if filter_type == 'hilbert':
        # Use the hilbert filter defined as 
        # f(k_x) =  1i/2pi for k_x < 0
        #          -1i/2pi for k_x > 0
        # or multiply the ramp the fourier integration kernel
        
        f = np.zeros(q.shape, dtype=np.complex128)
        f[1:N//2+1]  = -1j / 2. / np.pi
        f[N//2+1:] = 1j / 2. / np.pi
        # or
        f = np.asarray(q, dtype=np.complex)
        f[np.real(f)<0] =  1j / (2.*np.pi)
        f[np.real(f)>0] = -1j / (2.*np.pi)        
        # or
        f = np.abs(q)/(1j*2.*np.pi*q)
        f[0] = 0.
    elif filter_type == 'hamming':
        f = np.abs(q)  # ramp filter
        f = f * (0.54 + 0.46 * np.cos(2 * np.pi * q))

    # Filter sinogram in Fourier space in detector direction
    sino_ft = np.fft.fft(sinogram)  # By default along last axis
    sino_ft *= f
    sino_filtered = np.real(np.fft.ifft(sino_ft))  # By default along last axis

    # Backprojection 
    recon = np.zeros((N, N))

    for i, angle in enumerate(angle_list):
        proj_spread = np.tile(sino_filtered[i, :], (N, 1))
        recon += nd.interpolation.rotate(
            proj_spread, angle, reshape=False, order=1)
    
    # Normalize and crop border
    recon /= Nangles
    recon = apply_mask(recon)    
    
    return recon

# Read in the absorption and differential phase contrast sinograms
sinogram_abs = np.load('sinogram_amp.npy').T
sinogram_dpc = np.load('sinogram_dpc.npy').T
Nangles, N = sinogram_abs.shape

# Integrate the dpc projections along the detector direction to acquire
# non-differential phase contrast projections. 
# Sanity check: If you differentiate the result you should get back the 
# original dpc projections
# Integration can be done in various ways, for example, by multiplication with 
# the kernel 1/(2pi*k*i) in fourier space or by a cumulative sum over the array
sinogram_pc = np.cumsum(sinogram_dpc, axis=-1)
sinogram_pc -= np.mean(sinogram_pc,axis=-1)[:,None]
# or
sinogram_dpc_fft = np.fft.fft(sinogram_dpc, axis=-1)
fft_integration_kernel = 1. / (2.*np.pi*np.fft.fftfreq(N) * 1j)
fft_integration_kernel[0] = 0.
sinogram_dpc_fft *= fft_integration_kernel
sinogram_pc = np.real(np.fft.ifft(sinogram_dpc_fft, axis=-1))

# Plot the sinogram of the absorption and the differential phase contrast.
plt.figure(1, figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('absorption')
plt.imshow(sinogram_abs, cmap='gray')
plt.colorbar()
plt.subplot(1, 3, 2)
plt.title('differential phase')
plt.imshow(sinogram_dpc, cmap='gray')
plt.colorbar()
plt.subplot(1, 3, 3)
plt.title('Phase')
plt.imshow(sinogram_pc, cmap='gray')
plt.colorbar()


# Perform filtered backprojection of the absorption using the ramp filter
fbp_attenuation = fbp(sinogram_abs, 'ramp')


# Perform filtered backprojection of the dpc projections using Hilbert filter
# and of the integrated dpc projections using the ramp filter
fbp_refraction_hilb = fbp(sinogram_dpc, 'hilbert')
fbp_refraction_ramp = fbp(sinogram_pc, 'ramp')


# Plot the reconstructions of refractive index decrement and attenuation coef.
plt.figure(2, figsize=(6, 12))
plt.subplot(3, 1, 1)
plt.title('Linear attenuation coefficient')
plt.imshow(fbp_attenuation, cmap='gray')
plt.colorbar()
plt.subplot(3, 1, 2)
plt.title('Refr. ind. decr. - Hilbert Filter')
plt.imshow(fbp_refraction_hilb, cmap='gray')
plt.colorbar()
plt.subplot(3, 1, 3)
plt.title('Refr. ind. decr. - Integration+Ramp Filter')
plt.imshow(fbp_refraction_ramp, cmap='gray')
plt.colorbar()

