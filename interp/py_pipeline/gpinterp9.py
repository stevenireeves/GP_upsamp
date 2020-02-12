import numpy as np
import scipy.misc as sp
import cv2
from matplotlib import pyplot as plt

import kernels
import weights
import interp

#img = cv2.imread('bad_receipts/2867_ffb541fd-392e-49af-b139-f179f6bf114d.jpg') 
#img = cv2.imread('bad_receipts/2838_6893a1e4-1f38-49e2-a14c-5e0359623d4f.jpg')
img = np.ones((100,100,3))
#img = cv2.imread('bad_receipts/2937_6436a055-0c76-4f99-a375-9f4f43e33bfd.jpg')
hc, wc = img.shape[:2]
#GP Parameters
dx = 1. / wc
dy = 1. / hc
l = 12*min(dx,dy)
lx = 2
ly = 2

pnt = weights.get_pnt9(dx, dy)
kpnt = weights.get_kpnt(dx,dy,lx)
K = weights.get_k(pnt, l, 9)
Kdet = weights.get_k(pnt, 3*min(dx,dy), 9)
eig, vec = np.linalg.eigh(Kdet)
Kinv = np.linalg.inv(K) 
kp = weights.get_kpnt(dx,dy, lx)
ks = weights.get_ks(kpnt, pnt, l, 9, lx)
for idy in range(lx*ly):
    ks[:,idy] = np.dot(ks[:, idy], Kinv)

#resolution for finely interpolated image
hf = hc * ly
wf = wc * lx
img2 = np.zeros((hf, wf, 3))
img = img/255.0
interp.gp9(wc, hc, lx, ly, ks, img, img2)
img2 = img2*255.0
print(img2)
cv2.imwrite('testgp9.jpg', img2)
