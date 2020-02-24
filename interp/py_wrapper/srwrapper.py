import ctypes
import os 

import numpy as np 
import numpy.ctypeslib as npct

def GPupsamp(img_in, ratio, weights, mle):
    try:
        SO_PATH = os.environ['SO_PATH']
        libinterp = ctypes.cdll.LoadLibrary(SO_PATH)
    except Exception:
        print("Path to C++ function not set! Please use export SO_PATH=<path/to/C++lib>")
        print("Exiting")
        quit()
    interpolate = libinterp.interpolate
    interpolate.restype = None
    interpolate.argtypes = [npct.ndpointer(ctypes.c_ubyte), npct.ndpointer(ctypes.c_float),
                            npct.ndpointer(ctypes.c_int), npct.ndpointer(ctypes.c_int), 
                            npct.ndpointer(ctypes.c_float), npct.ndpointer(ctypes.c_float)]
    size = np.asarray(img_in.shape, dtype=np.int32)
    upsamp = np.array([ratio, ratio],dtype=np.int32)
    img_out = np.zeros((size[0]*upsamp[0], size[1]*upsamp[1]), dtype=np.float32)
    interpolate(img_in, img_out, upsamp, np.array([size[0], size[1]], dtype=np.int32), weights, mle)
    return img_out
