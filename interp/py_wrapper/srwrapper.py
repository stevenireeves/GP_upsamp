import ctypes
import numpy as np 
import numpy.ctypeslib as npct
import cv2 
import time

libinterp = ctypes.cdll.LoadLibrary('../cpp_pipeline/interp.dylib')
filename = input('Enter a filename ')
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
interpolate = libinterp.interpolate
interpolate.restype = None
interpolate.argtypes = [npct.ndpointer(ctypes.c_float), npct.ndpointer(ctypes.c_float),
                        npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int)]
size = np.asarray(img.shape, dtype=np.int32)
upsamp = np.array([4, 4],dtype=np.int32)
img = img/255.0
img = img.flatten().astype(np.float32)
img_out = np.zeros(size[0]*size[1]*upsamp[0]*upsamp[1], dtype=np.float32)
start = time.time()
interpolate(img, img_out, upsamp, np.array([size[1], size[0]], dtype=np.int32))
stop = time.time()
print(stop - start)
img_out = img_out*255
img_out = img_out.reshape((size[0]*upsamp[0], size[1]*upsamp[1]))
#img_out = img_out.astype(np.uint8)
cv2.imwrite('img_out.jpg', img_out)
