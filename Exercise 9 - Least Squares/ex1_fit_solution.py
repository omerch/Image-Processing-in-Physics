"""
13.07.2017
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Maximilian Teuffenbach)

ex1_fit.py

Script to perform linear least squares data fitting to a set of experimental
data. Some other important information:
* The signal is periodic with a frequency f of 1.
* Because of some mis-calibration of your measurement setup, the measured
  signal is 'trended': that is there is a constant offset, and a linear
  dependance of the signal
* i.e. measurements = constant + a*x + b*sin(f*x)

As per usual, replace the ???s with the appropriate command(s).
"""
import numpy as np
import matplotlib.pyplot as plt

# Read in the date from the file data.npy

exp_data = np.load('data.npy')

# The first column of exp_dat is x (sample points)

x = exp_data[:, 0]

# The second column of exp_data is measurements

measurements = exp_data[:, 1]

# Display the measured data

plt.figure(1)
plt.plot(x, measurements, 'o')
plt.title('Measured data points')
plt.xlabel('x')
plt.ylabel('measurement')


# We want to make a least squares fit of a constant, linear, and sine

# The linear part is a multiple of the x values
mat_linear = x

# Create the first column of the matrix for the constant offset
mat_const = np.ones(len(x))

# Create the second column of the matrix for the sinusoid (frequency is 1)
f = 1
mat_sine = np.sin(f*x)

# Use np.vstack to combine the three columns into a matrix
A = np.vstack((mat_const, mat_linear, mat_sine))

# A has still the wrong orientation so we take the transpose of it
A = A.T

# EITHER use the equation for general linear least squares from the lecture
# Hint: np.dot for matrix multiplies
# Hint: np.linalg.inv for matrix inverse
# Hint: np.T for matrix transpose
coeff = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), measurements)

# AND/OR use the built in function np.linalg.lstsq (watch output parameters)
coeff2 = np.linalg.lstsq(A, measurements)[0]

# The two methods should give exactly the same coefficients !!!


# Calculate an array of regularly spaced values to use to calculate the fitted
# function. Hint: use np.linspace
x_fit = np.linspace(x.min(), x.max(), 300)

# Calculate the least squares fit from the coeff and x_fit
y_fit = coeff[0] + coeff[1]*x_fit + coeff[2]*np.sin(f*x_fit)

# Plot data and the least squares fit
plt.figure(2)
plt.title('Measurememts & Least Squares fit')
plt.plot(x, measurements, 'o', label='Measurements')
plt.plot(x_fit, y_fit, label='Least Squares fit')
plt.legend()
plt.xlabel('x')
plt.ylabel('measurement')
