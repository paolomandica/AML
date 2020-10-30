# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2, convolve
from scipy.ndimage import gaussian_filter1d


"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""


def gauss(sigma):

    # create array of integer values x
    x = np.arange(int(-3*sigma), int(3*sigma+1))
    # Compute Gaussian
    Gx = (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-x**2 / (2*sigma**2))

    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""


def gaussianfilter(img, sigma):
    kernel, _ = gauss(sigma)

    # Compute Gaussian on rows
    partial = np.apply_along_axis(convolve, 0, img, kernel, mode="same")
    # Compute Gaussian on columns
    smooth_img = np.apply_along_axis(convolve, 1, partial, kernel, mode="same")

    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""


def gaussdx(sigma):

    # Compute gaussian filter
    Gx, x = gauss(sigma)
    # Derive gaussian filter
    Dx = np.gradient(Gx)

    ### We also tried a manual implementation of the Gaussian derivative
    ### filter and, since the result is the same, we kept the numpy version.
    # x = np.arange(int(-3*sigma), int(3*sigma+1))
    # Dx = -1/(np.sqrt(2*np.pi)*sigma**3)*x*np.exp(-x**2/ (2 * sigma**2))

    return Dx, x


def gaussderiv(img, sigma):

    kernel, _ = gaussdx(sigma)
    # Compute Gaussian on rows
    imgDx = np.apply_along_axis(convolve, 0, img, kernel, mode="same")
    # Compute Gaussian on columns
    imgDy = np.apply_along_axis(convolve, 1, img, kernel, mode="same")

    return imgDx, imgDy
