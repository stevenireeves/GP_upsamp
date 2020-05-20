'''
This module contains the functions needed to generate the model weights
for the GP upsampling method. These weights are passed to the C++
framework to generate the upsampled document image.

Using this module allows for the C++ framework to be built without
the libraries LAPACKE and OpenBlas.
'''
import numpy as np

from . import kernels

def get_k(pnt, lscale, size):
    '''
    :inputs: Pixel locations, length scale, size of sample
    :outputs: The Covariance Kernel over the sample 5x5 pixel window
    '''
    kov = np.zeros((size, size))
    for idx in range(size):
        for idy in range(size):
            kov[idx, idy] = kernels.matern3(pnt[idx], pnt[idy], lscale)
    return kov


def get_ks(pnt1, pnt2, lscale, sz1, refratio):
    '''
    :inputs: Prediction Pixel Locations, Sample Pixel Locations, length scale,
             sample size (5x5 = 25), and upsampling factor
    :outputs: The covariance vectors for each predicted pixel relative to the sample
    '''
    kovs = np.zeros((sz1, refratio**2))
    for idy in range(refratio**2):
        for idx in range(sz1):
            kovs[idx, idy] = kernels.matern3(pnt1[idy], pnt2[idx], lscale)
    return kovs


def get_pnt(delx, dely):
    '''
    :inputs: inverse pixel size for cols and rows
    :outputs: Pixel locations for 5x5 window
    '''
    grdx, grdy = np.mgrid[-2:3, -2:3]
    grdx = grdx * delx
    grdy = grdy * dely
    pnts = np.array([grdx.flatten(), grdy.flatten()]).transpose()
    return pnts


def get_kpnt(delx, dely, refratio):
    '''
    :inputs: inverse pixel size for cols and rows, and upsampling factor
    :outputs: Predicted pixel locations
    '''
    kpnt = np.zeros((refratio**2, 2))
    if refratio == 4:
        start = -0.375
    else:
        start = -0.25
    for idx in range(refratio):
        for idy in range(refratio):
            idk = idx * refratio + idy
            kpnt[idk, 0] = (start + 1.0 / refratio * idx) * delx
            kpnt[idk, 1] = (start + 1.0 / refratio * idy) * dely
    return kpnt


def get_weights(img_size, refratio, lscale):
    '''
    :intputs: image size, upsampling factor, and length scale
              img_size    refratio               lscale
    :output: Gaussian Process Model weights for the 5x5 window
             w*  = k*^T K^{-1}
             to be used with all 5x5 windows, output size refratio**2*5*5
             and the maximum likelihood estimate weights mle_v
    '''
    delx = 1 / img_size[0]  #img_size[1] = # of cols -> x dim
    dely = 1 / img_size[1]  #img_size[0] = # of rows -> y dim
    pnts = get_pnt(delx, dely)
    #Build covariance Kernel over Sample 5x5 pixel window
    ell = lscale * min(delx, dely)
    kov_m = get_k(pnts, ell, 25)
    #Build convaraince vectors in relation to sample window and predicted pixel locations
    kpnts = get_kpnt(delx, dely, refratio)
    kovs_v = get_ks(kpnts, pnts, ell, 25, refratio)
    mle_v = np.linalg.solve(kov_m, np.ones(25))
    mle_v /= np.sum(mle_v)
    return np.float32(np.linalg.solve(kov_m, kovs_v)), np.float32(mle_v)
