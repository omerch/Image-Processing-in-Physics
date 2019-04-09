# -*- coding: utf-8 -*-
"""
Image Processing in Physics Toolbox

This toolbox is a collection of the functions that we will program in the 
course of this lecture, and that we will collect in this module for future use.

18.11.2015
Image Processing Physics TU Muenchen
Julia Herzen, Peter B. Noel, Kaye Morgan, (Sebastian Allner)

"""

from PIL import Image as pil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import threading
import pywt #can be downloaded for free from http://www.pybytes.com/pywavelets/

__all__ = ['dwt_multiscale', 'idwt_multiscale', 'tile_dwt', 'rescale']


def dwt_multiscale(image, nLevel=3, wavelet='db1', mode='cpd'):
    """Calculate a multilevel 2D discrete wavelet transform"""    
    A = np.array([])
    H = np.array([])
    V = np.array([])
    D = np.array([])
    coeffs = []
    #initialize image variable with input image
    im=image
    #perform multilevel decomposition
    for iLevel in range(nLevel):
        #perform wavelet transform of image variable        
        appr, (hori,vert,diag) = pywt.dwt2(im,wavelet,mode)
        #save coefficient results
        A = np.hstack([A, appr.ravel()])
        H = np.hstack([H, hori.ravel()])
        V = np.hstack([V, vert.ravel()])
        D = np.hstack([D, diag.ravel()])
        coeffs.append((appr,hori,vert,diag))
        # save approximation aa at level iLevel in image variable
        im = appr
    return coeffs, (A,H,V,D)


def idwt_multiscale(coeffs, wavelet='db1', mode='cpd'):
    """Calculate a multilevel 2D inverse discrete wavelet transform"""    
    recons = coeffs[-1][0]
    nLevel = len(coeffs)
    for iLevel in reversed(range(nLevel)):
        recons = pywt.idwt2( (recons, tuple(coeffs[iLevel][1:])), wavelet, mode)
    return recons


def tile_dwt(coeffs, shape):
    """Helper routine to tile the 2D-wavelet coefficients into the standard shape"""
    tiled_image = np.zeros(shape)
    
    # add n-th level approximation into corner of tiled image
    A0 = coeffs[-1][0]
    tiled_image[0:A0.shape[0],0:A0.shape[1]] = rescale(A0, (0,1))
    
    nLevel = len(coeffs)
    for iLevel in reversed(range(nLevel)):
        # read coefficients at level iLevel
        appr,hori,vert,diag = coeffs[iLevel]
        Vert=rescale(abs(vert), (0,1))
        Hori=rescale(abs(hori), (0,1))
        Diag=rescale(abs(diag), (0,1))
        #determine shape parameters at level iLevel
        i0 = int(np.floor( shape[0] * .5**(iLevel+1)) )
        j0 = int(np.floor( shape[1] * .5**(iLevel+1)) )
        ir, jr = appr.shape
        # tile subimages at level iLevel
        tiled_image[i0:(i0+ir),0:jr]       = Vert
        tiled_image[0:ir,j0:(j0+jr)]       = Hori
        tiled_image[i0:(i0+ir),j0:(j0+jr)] = Diag                    
    return tiled_image


def rescale(a, bounds):
    """ Helper routine for linear rescaling of the input onto a interval 
        given by "bounds" """
    b0,b1 = (min(bounds), max(bounds))
    a_scaled= b0 + (b1-b0)*(a.astype('float')-a.min())/(a.max()-a.min())
    return a_scaled
