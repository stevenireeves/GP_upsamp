import numpy as np
import scipy.misc as sp
from kernels import * 
import cv2
from matplotlib import pyplot as plt 

def gp(sten, ksid):
	global Kinv, ks
	one = np.ones(5)
	iterp = np.dot(ks[:,ksid], sten)
	return iterp

def interp(width, height, lx, ly, dat_in, dat_out):
#3 Channels
	for k in range(3):
#Need to think of border cases 
		for i in range(1,height-1):
			for j in range(1,width-1):
				sten = np.array([dat_in[i,j-1,k], dat_in[i-1,j,k], dat_in[i,j,k], dat_in[i+1,j,k], dat_in[i,j+1,k]])
				for iref in range(lx):	
					ii = i*lx + iref
					for jref in range(ly): 
						jj = j*ly + jref
						ksid = iref + jref*lx 
						dat_out[ii,jj,k] = gp(sten, ksid)
#Just copy border for now
		for i in range(0, height, height-1):
			for j in range(0, width):
				for iref in range(lx):
					ii = i*lx + iref
					for jref in range(ly):
						jj = j*ly + jref
						dat_out[ii,jj,k] = dat_in[i,j,k]

		for i in range(0, height):
			for j in range(0, width, width-1):
				for iref in range(lx):
					ii = i*lx + iref
					for jref in range(ly):
						jj = j*ly + jref
						dat_out[ii,jj,k] = dat_in[i,j,k]


#Main Program
img = cv2.imread('../../Caltech256/256_ObjectCategories/207.swan/207_0001.jpg')
img = img/255 
hc, wc = img.shape[:2]
#GP Parameters 
dx = 1./wc 
dy = 1./hc 
l = 12*min(dx,dy) 
lx = 4 
ly = 4

#Global Vars 
#K = np.eye(5) 

#Building covariance matrix 
#a = np.exp(-(dx**2 + dy**2)/(2*l**2))
#b = np.exp(-dy**2/(2*l**2))
#c = np.exp(-4*dy**2/(2*l**2))
#d = np.exp(-dx**2/(2*l**2))
#e = np.exp(-4*dx**2/(2*l**2))
#K[0,1:] = [a, b, a, c] 
#K[1,2:] = [d, e, a]
#K[2,3:] = [d, b] 
#K[3,4] = a
#K[1:,0] = K[0,1:]
#K[2:,1] = K[1,2:]
#K[3:,2] = K[2,3:]
#K[4,3] = K[3,4] 
#Kinv = np.linalg.inv(K) 

#build covariance Vectors
pnt =np.array([[ 0, -dy]
			,[-dx,  0]
			,[ 0,  0]
			,[ dx,  0]
			,[ 0,  dy]])
K = np.zeros((5,5))
for i in range(5): 
	for j in range(5): 
#		K[i,j] = matern5(pnt[i], pnt[j], l)
		K[i,j] = matern3(pnt[i], pnt[j], l) 
#		K[i,j] = sqrexp(pnt[i], pnt[j], l)

Kinv = np.linalg.inv(K) 
kp = np.zeros((lx*ly, 2))
start = -0.25 
if lx == 4: 
	start = -0.375 
for j in range(ly):
	for i in range(lx):  
		idk = i + lx*j
		kp[idk, 0] = (start + 1.0/lx*i)*dx
		kp[idk, 1] = (start + 1.0/ly*j)*dy

ks = np.zeros((5,lx*ly))
for idy in range(lx*ly): 
	for idx in range(5): 
#		ks[idx, idy] = sqrexp(kp[idy, :], pnt[idx][:],l)
#		ks[idx, idy] = matern5(kp[idy, :], pnt[idx][:], l) 
		ks[idx, idy] = matern3(kp[idy, :], pnt[idx][:], l) 


for i in range(lx*ly): 
	ks[:,i] = np.dot(ks[:,i], Kinv)

#resolution for finely interpolated image 
hf = hc*ly
wf = wc*lx
#img1 = sp.imresize(img, [hf,wf,3], 'bilinear', mode=None) 
img2 = np.zeros((hf,wf,3)) 

interp(wc, hc, lx, ly, img, img2)
img2 = img2*255
#if lx==2:
#tru = cv2.imread('images/superbike_pixabay.jpg')
#ht, wt = tru.shape[:2]
#if ht != hf or wt != wf: 
#	truesize = sp.imresize(tru, [hf, wf, 3], mode=None) 
#else:
#	truesize = tru
#err = abs(img1 - truesize)
#cv2.imwrite('true_resize.jpg', truesize)
#cv2.imwrite('err5blin.jpg',err)
#cv2.imwrite('interpblin2.jpg', img1) 
cv2.imwrite('swan_gp.jpg',img2)
