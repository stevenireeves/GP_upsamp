import numpy as np
import cv2
from .weights import get_weights
import matplotlib.pyplot as plt

def gp_conv_sr(img, ratio, ell):
    ks, c = get_weights(img.shape[0:2], ratio, ell)
    ks = np.swapaxes(ks,0,1)
    for i in range(ks.shape[0]):
        ks[i] -= ks[i].sum(dtype=np.float32)*c
    c = c.reshape((5, 5))
    ks = ks.reshape((ks.shape[0], 5, 5))
    mle = cv2.filter2D(np.float32(img), -1, c)
    filters = np.zeros((img.shape[0], img.shape[1], ratio*ratio), dtype=np.float32)
    for i in range(ks.shape[0]):
        filters[:,:,i] = mle + cv2.filter2D(np.float32(img),-1, ks[i])#, anchor=(3,3))
    filters = filters.reshape(filters.shape[0], filters.shape[1], ratio, ratio)
    filters = filters.transpose((0,2,1,3))
    return filters.reshape(img.shape[0]*ratio, img.shape[1]*ratio)

