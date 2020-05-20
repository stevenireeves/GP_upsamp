'''
This module contains the Matern3/2 Covariance function to build the
GP model.
'''
import numpy as np

def matern3(xloc, yloc, rho):
    '''
    :inputs: xloc, yloc - pixel locations,
             rho - characteristic length scale
    :outputs: Matern 3/2 covariance between xloc and yloc
    '''
    dis = np.linalg.norm(xloc-yloc)
    arg = np.sqrt(3)*(dis/rho)
    return (1 + arg)*np.exp(-arg)
