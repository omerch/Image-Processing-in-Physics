"""
01.06.2017
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Maximilian Teuffenbach)

ex1_correlation.py

Using numpy, matplotlib, scipy

The goal of this exercise is to get acquainted with noise, and the related
concepts of noise power spectra and correlation.
The exercise is split into two short subtasks.
You need to replace the ??? in the code with the required commands.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

###############################################################################
# Part A: noise and correlation
# create noise, and calculate its noise power spectrum and correlation
###############################################################################

# Create a 100x100 array of Gaussian noise with mean=0 and standard deviation
# sigma = 1. Use the function numpy.rand.randn.
# Then use scipy.ndimage.gaussian_filter to create a low and high pass
# filtered version of your noise. (Remember from lecture: a high pass can be
# modelled as the original image minus the low pass image)

mean = 0
sigma = 1
M = 100
white_noise = mean + sigma * np.random.randn(M, M)
low_pass = nd.gaussian_filter(white_noise, sigma)
high_pass = white_noise - low_pass

# Calculate and plot the noise power spectra of each noise signal.

nps_white = np.fft.fftshift(np.abs(np.fft.fft2(white_noise)))**2
nps_low = np.fft.fftshift(np.abs(np.fft.fft2(low_pass)))**2
nps_high = np.fft.fftshift(np.abs(np.fft.fft2(high_pass)))**2

# Calculate and plot the auto-correlation of each noise signal using the
# correlation theorem. Center the maximum cross-correlation in the middle of
# the image, as already shown in the lecture for white noise.

corr_white = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(white_noise))**2)))
corr_low = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(low_pass))**2)))
corr_high = np.real(np.fft.fftshift(np.fft.ifft2(
    np.abs(np.fft.fft2(high_pass))**2)))

# Please note that this is NOT the same as
# corr_white = np.fft.fftshift(np.fft.ifft2(nps_white)).real
# corr_low = np.fft.fftshift(np.fft.ifft2(nps_low)).real
# corr_high = np.fft.fftshift(np.fft.ifft2(nps_high)).real
# Due to the additional fftshift incorporated in nps

# The autocorrelation tells you the correlation between two pixels as a
# function of the distance between those pixels. For white noise, the
# correlation is only nonzero if the distance is zero, else there is no
# correlation between pixels.
# For the low-pass noise, there is a correlation in the neighborhood of each
# pixel due to the "patchy" character of the noise. The correlation falls off
# quickly with increasing distance. The high-pass noise exhibits anti-
# correaltion in its immediate neighbourhood,i.e. you KNOW that a pixel will
# be different from its neighbor due to the "fast" changes in the noise.

##############################################################################
# Part B: Image shift using cross-correlation
# Use cross-correlation for a simple image registration task
##############################################################################

# Read in the two images worldA and worldB. Both images show the same object,
# but shifted by a small amount relative to each other. The task is to
# estimate the shift using cross-correlation.

im_shifted1 = plt.imread('worldA.jpg') / 255.
im_shifted2 = plt.imread('worldB.jpg') / 255.
im_shifted1 = im_shifted1.mean(axis=2)
im_shifted2 = im_shifted2.mean(axis=2)

# Calculate the cross-correlation between the two images

ccorr = np.real(np.fft.ifftshift(np.fft.ifft2(
    np.conj(np.fft.fft2(im_shifted1)) * np.fft.fft2(im_shifted2))))

# Calculate the shift as a vector tuple. (you might want to use numpy.argmax
# and np.unravel_index, or numpy.where)

shift_y, shift_x = np.unravel_index(ccorr.argmax(), ccorr.shape)

# Print to screen the shifts

print "shift in y = " + str(shift_y - ccorr.shape[0] / 2)
print "shift in x = " + str(shift_x - ccorr.shape[1] / 2)

# The crosscorrelation will be highest if the shift from the correlation
# equals the shift between the two images, so you have to search for the
# coordinate of the maximum cross-correlation relative to the origin.

##############################################################################
# Plot results
##############################################################################

# Part A

plt.figure(1, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(white_noise, cmap='gray', interpolation='none')
plt.title('white noise spatial domain')
plt.subplot(1, 3, 2)
plt.imshow(low_pass, cmap='gray', interpolation='none')
plt.title('low pass spatial domain')
plt.subplot(1, 3, 3)
plt.imshow(high_pass, cmap='gray', interpolation='none')
plt.title('high pass spatial domain')

plt.figure(2, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(nps_white, cmap='gray', interpolation='none')
plt.title('white noise power spectrum')
plt.subplot(1, 3, 2)
plt.imshow(nps_low, cmap='gray', interpolation='none')
plt.title('low pass power spectrum')
plt.subplot(1, 3, 3)
plt.imshow(nps_high, cmap='gray', interpolation='none')
plt.title('high pass power spectrum')

plt.figure(3, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(corr_white, cmap='gray', interpolation='none')
plt.title('white noise autocorrelation')
plt.subplot(1, 3, 2)
plt.imshow(corr_low, cmap='gray', interpolation='none')
plt.title('low pass noise autocorrelation')
plt.subplot(1, 3, 3)
plt.imshow(corr_high, cmap='gray', interpolation='none')
plt.title('high pass noise autocorrelation')

# Part B

plt.figure(4, figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(im_shifted1, cmap='gray', interpolation='none')
plt.title('image1')
plt.subplot(1, 3, 2)
plt.imshow(im_shifted2, cmap='gray', interpolation='none')
plt.title('image2')
plt.subplot(1, 3, 3)
plt.imshow(ccorr, cmap='gray', interpolation='none')
plt.title('crosscorrelation')
