"""
26.10.2016
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Lorenz Hehn)

ex2_laplace_filtering.py

Laplace filter in frequency domain

Your task in this exercise is to create your own implementation of a
Laplace filter in Fourier space and apply it to an image.
The formula for the Laplacian in the Fourier domain is:
    H(u,v) = -4*pi^2*(u^2+v^2)      source: (Gonzalez, chapter 4, p286)

You need to replace the ??? in the code with the required commands
"""

import numpy as np
import matplotlib.pyplot as plt

# Load venice.jpg using imread, normalize it to (0, 1)
# and take the red channel again
img = plt.imread('venice.jpg') / 255.
img = img[:, :, 0]

# Plot the image before applying the filter
plt.figure(1)
plt.imshow(img, cmap='gray')
plt.colorbar()

# Generate the coordinate systems v and u
# You can use the numpy function linspace (or meshgrid) to create them.
# Look up the documentation of linpace to get familiar with its parameters.
# Your v and u arrays should afterwards start at -0.5 and get to 0.5 in N
# steps, where N is defined by the image shape in the respective dimension
v = np.linspace(-.5, .5, img.shape[0])
u = np.linspace(-.5, .5, img.shape[1])

# the function np.meshgrid creates coordinate arrays for the v and the u
# coordinates and writes them into vv and uu
# you can display them with plt.figure(); plt.imshow(uu); colorbar() if you
# want to have a look at them
vv, uu = np.meshgrid(v, u, indexing='ij')

# Caluclate the filter function H(v, u)
# If you want to do this in one line use vv and uu, as they are both of the
# image shape. The formula is given in the very top documentation of this
# script. Check if H has the same shape as the image.
H = -4 * np.pi**2 * (vv**2 + uu**2)

# Calculate the Fourier transform of the image
# You can use the numpy function fft2 included in np.fft
img_ft = np.fft.fft2(img)

# Multiply the FT of the image by the filter function
# The potential function H is centered around the center of the image.
# But actually it should be centered around the top left corner of the image,
# because a Fourier transform in python always has the central frequencies in
# the top left corner. Therefore, you have to use an fftshift of H, which
# corrects for that. (It can be found in np.fft .)
# Check out the looks of both H and np.fft.fftshift(H) with plt.imshow(???),
# to understand what it does.

# Take the inverse Fourier transform of the product to get the filtered image
# and select the real part of it, as we do not want to have the imaginary part
# of real images.

img_filtered = np.real(np.fft.ifft2(img_ft * np.fft.fftshift(H)))

plt.figure(2)
plt.imshow(img_filtered, cmap='gray')
plt.colorbar()
