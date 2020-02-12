import numpy as np
import scipy.misc as sp
import cv2
from matplotlib import pyplot as plt 


def downsample(width, height, lx, ly, dat_in, dat_out):
#3 Channels
	for k in range(3):
#Need to think of border cases 
		for i in range(height):
			for j in range(width):
				for iref in range(lx):	
					ii = i*lx + iref
					for jref in range(ly): 
						jj = j*ly + jref
						dat_out[i,j,k] += dat_in[ii,jj,k]
				dat_out[i,j,k] /= 4; 
#Main Program
img = cv2.imread('images/superbike_pixabay.jpg')
hc, wc = img.shape[:2]
hc //= 2
wc //= 2
lx = 2
ly = 2

img2 = np.zeros((hc,wc,3)) 

downsample(wc, hc, lx, ly, img, img2)
cv2.imwrite('sbk_down.jpg',img2)
