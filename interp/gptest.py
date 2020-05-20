import numpy as np 
import matplotlib.pyplot as plt 
import cv2
from py_pipeline.weights import get_weights
from py_pipeline.gp_conv_sr import gp_conv_sr
from py_wrapper.srwrapper import GPupsamp

filename = input('Enter a filename ')
ratio = int(input('Enter an upsampling ratio: 2 or 4 '))
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#ks, c = get_weights(img.shape[0:2], ratio, 20.)
img_out = gp_conv_sr(img, ratio, 12.)

#img_out = GPupsamp(img, ratio, ks, c)
cv2.imwrite('img_out.png', img_out)
