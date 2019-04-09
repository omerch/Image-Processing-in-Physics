"""
08.06.2017
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Maximilian Teuffenbach)

ex1_segmentation.py

This exercise is all about counting stars.
The goal is to know how many stars are in the image and what sizes they are.
Problematic is only the moon.

As per usual, replace the ???s with the appropriate command(s).
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# Load the respective image

img = plt.imread('stars.jpg')

# Sum up all color channels to get a grayscale image.
# use numpy function sum and sum along axis 2, be careful with the datatypes
# rescale the finale image to [0.0 1.0]

img = np.sum(img, axis=2, dtype=float)
img = img / img.max()

# Now look at your image using imshow. Use vmin and vmax parameters in imshow

plt.figure(1)
plt.title('img')
plt.imshow(img, cmap='gray', interpolation='none', vmin=0., vmax=1.)
plt.colorbar()

# You can set thresholds to cut the background noise
# Once you are sure you have all stars included use a binary threshold.
# (Tip: a threshold of 0.1 seemed to be good, but pick your own)

threshold = 0.1
img_bin = img > threshold

plt.figure(2)
plt.title('img_bin')
plt.imshow(img_bin, cmap='gray', interpolation='none')

# Now with the binary image use the opening and closing to bring the star
# to compacter format. Take care that no star connects to another

s1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
img_bin1 = nd.binary_closing(img_bin, structure=s1)

plt.figure(3)
plt.title('img_bin1')
plt.imshow(img_bin1, cmap='gray', interpolation='none')

# Remove isolated pixels around the moon with closing by a 2 pixel structure

s2 = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
img_bin2 = nd.binary_opening(img_bin1, structure=s2)

plt.figure(4)
plt.title('img_bin2')
plt.imshow(img_bin2, cmap='gray', interpolation='none')

# play with all the morphological options in ndimage package to increase the
# quality if still needed

s3 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
img_bin3 = nd.binary_opening(img_bin2, structure=s3)  # optional

plt.figure(5)
plt.title('img_bin3')
plt.imshow(img_bin3, cmap='gray', interpolation='none')

# plotting the sum of all your binary images can help identify if you loose
# stars. In principal every star is present in every binary image, so real
# stars have always at least one pixel maximum

plt.figure(6)
plt.imshow((np.int64(img_bin) + np.int64(img_bin1) + np.int64(img_bin2) +
           np.int64(img_bin3)), interpolation='none')
plt.colorbar()

# Once you're done, label your image with nd.label

img_lbld, num_stars = nd.label(img_bin3)

plt.figure(7)
plt.imshow(img_lbld, interpolation='none')
plt.colorbar()

# Use nd.find_objects to return a list of slices through the image for each
# star

slc_lst = nd.find_objects(img_lbld)

# You can have a look now at the individual stars. Just apply the slice to your
# labelled array

starnum = 105

plt.figure(8)
plt.title("star %i" % starnum)
plt.imshow(img_lbld[slc_lst[starnum-1]], cmap='gray', interpolation='none')

# REMAINING task. Sum up each individual star to get a list of star
# brightnesses
# make a detailed histogram (>100 bins). Take care to exclude the moon!
# This can be done by sorting the brightness array and remove the last element.

# Remember: im_lbld[slc_lst[<number>]] selects one star. Create a list of star
# images (star_lst) with [0, 1]. I help you with that.
# Afterwards, sum their intensity up (take care about the datatypes),
# and np.sort the sums.

star_lst = [img_lbld[slc] > 0 for slc in slc_lst]
mass_lst = [np.sum(np.int64(star)) for star in star_lst]
mass_lst_sorted = np.sort(mass_lst)

plt.figure(9)
plt.title("brigtnesses of stars")
plt.hist(mass_lst_sorted[:-1], bins=200, range=(0, 200), align='left')
plt.xlim([0, 200])
