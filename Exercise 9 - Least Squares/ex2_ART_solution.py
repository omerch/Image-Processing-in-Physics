"""
13.07.2017
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Maximilian Teuffenbach)

ex2_ART.py

Tomographic reconstruction with an algebraic reconstruction technique (ART)

In this exercise you will create your own implementation of ART
and do a reconstruction of the Shepp-Logan head phantom.

I provide functions in a separate module called tomolib.
Contained in there are the apply_mask, the backproject and the fbp functions.
The sinogram is given in phantom_sino256.npy
The phantom is given in phantom_tomo256.npy
As always, replace the ???s with the correct commands
For this exercise, make an initial guess for the tomogram of
all zeros. You can also try using an initial guess with the filtered
back-projection instead.
"""
import numpy as np
import matplotlib.pyplot as plt
import tomolib_solution as tomolib

# Load the sinogram
sinogram = np.load('phantom_sino256.npy')

# Plot the sinogram
plt.figure(1)
plt.clf()
plt.imshow(sinogram, cmap='gray', interpolation='none')
plt.colorbar()
plt.xlabel('Projection angles')
plt.ylabel('Spatial axis')

# Load the phantom. Your estimate of the tomogram should look something like
# this
phantom = np.load('phantom_tomo256.npy')

# Plot the phantom
plt.figure(2)
plt.clf()
plt.imshow(phantom, cmap='gray', interpolation='none')
plt.colorbar()
plt.title('Phantom')

# The number of iterations for the ART algorithm
Niter = 10

# Calculate the number of pixels (N) and angles (Nangles) from the sinogram
# shape
N, Nangles = sinogram.shape

# Create Nangles equally spaced theta values over the full 360 degrees
theta_list = 360. * np.arange(Nangles)/Nangles

# Start with an initial guess of zeros for the tomogram
initial_tomo = np.zeros((N, N))
# initial_tomo = tomolib.fbp(sinogram, filter_type='ramp')

# Prepare mask and renormalization term (Don't worry about this)
x, y = np.ogrid[-1:1:1j*N, -1:1:1j*N]
mask = (x*x + y*y <= 1.)
mask_proj = np.sum(mask, axis=0)
renorm = np.zeros(N)
renorm[mask_proj != 0] = 1. / mask_proj[mask_proj != 0]
# renorm[mask_proj == 0] = 0.

# Initialize
tomo = initial_tomo.copy()
# tomo = tomolib.fbp(sinogram)
error = []

# Main loop over the iterations
for i in range(Niter):
    err = 0
    plt.figure(3)
    plt.clf()
    plt.imshow(tomo, cmap='gray', interpolation='none')
    plt.colorbar()
    plt.title('ART reconstruction - after %i iterations' % i)
    plt.pause(0.1)

    # Loop over angles in the sinogram
    for index, th in enumerate(theta_list):

        # Forward-project your tomorgam for the current angle
        proj = tomolib.forwardproject(tomo, th)

        # Calculate the difference between the forward projection and
        # the sinogram for the current angle
        diff = sinogram[:, index] - proj

        # Accumulate the error to the total error
        err += (diff**2).sum()

        # Back-project the renormalized difference for the current angle
        # Hint - multiply the difference by renorm calculated above
        bpj = tomolib.backproject(renorm * diff, th)

        # Update the tomogram
        tomo += bpj

    print('Iteration ' + str(i) + ' completed.')
    print('Error: ' + str(err))
    error.append(err)

# Plot your estimate of the tomogram
plt.figure(3)
plt.clf()
plt.imshow(tomo, cmap='gray')
plt.colorbar()
plt.title('ART reconstruction - 10 iterations')

# Plot the error versus the number of iterations.
plt.figure(4)
plt.clf()
plt.plot(error, label='ART')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
