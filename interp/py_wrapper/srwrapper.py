import ctypes
import numpy as np 
import numpy.ctypeslib as npct

def GPupsamp(img_in, ratio):
    libinterp = ctypes.cdll.LoadLibrary('../cpp_pipeline/interp.dylib')
    interpolate = libinterp.interpolate
    interpolate.restype = None
    interpolate.argtypes = [npct.ndpointer(ctypes.c_float), npct.ndpointer(ctypes.c_float),
                            npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int)]
    size = np.asarray(img.shape, dtype=np.int32)
    upsamp = np.array([ratio, ratio],dtype=np.int32)
    img = img.flatten().astype(np.float32)
    img_out = np.zeros(size[0]*size[1]*upsamp[0]*upsamp[1], dtype=np.float32)
    interpolate(img, img_out, upsamp, np.array([size[1], size[0]], dtype=np.int32))
    img_out = img_out.reshape((size[0]*upsamp[0], size[1]*upsamp[1]))

