import numpy as np
import scipy.misc as sp
import cv2
from matplotlib import pyplot as plt

import kernels
import weights
import interp

#img = cv2.imread('bad_receipts/2867_ffb541fd-392e-49af-b139-f179f6bf114d.jpg', cv2.IMREAD_GRAYSCALE) 
#img = cv2.imread('bad_receipts/2838_6893a1e4-1f38-49e2-a14c-5e0359623d4f.jpg', cv2.IMREAD_GRAYSCALE)

#img = cv2.imread('bad_receipts/2937_6436a055-0c76-4f99-a375-9f4f43e33bfd.jpg')
#img = cv2.imread('/Users/stevenreeves/SISR4OCR/data/down_gray4/40_nouvel-obs_hbhnr300_constructedPdf_Nouvelobs2402PDF.orig.png', cv2.IMREAD_GRAYSCALE)
img = 255*np.ones((100,100))
hc, wc = img.shape[:2]
img = img[:,:,np.newaxis]#.astype(np.float32)
try:
    channels = img.shape[2]
except:
    channels = 1
#GP Parameters
dx = 1. / wc
dy = 1. / hc
l = 12*min(dx,dy)
lx = 4
ly = 4

pnt = weights.get_pnt9(dx, dy)
Kdet = weights.get_k(pnt, 3*min(dx,dy), 9)
eig, vec = np.linalg.eigh(Kdet)
ks = weights.get_MSweights(dx,dy, l, lx)
ktot = weights.get_SuperWeights(dx, dy, l, lx)
gamma = weights.get_gammas(ks, ktot, lx)

#resolution for finely interpolated image
hf = hc * ly
wf = wc * lx
img2 = np.zeros((hf, wf, channels), dtype=np.float32)
img = img/255.0
interp.multi_sampled_gp(wc, hc, lx, ly, ks, img, img2, vec, eig, gamma, channels)
img2 = img2*255.0
cv2.imwrite('GP_2867.png', img2)
