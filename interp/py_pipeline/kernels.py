import numpy as np 


# five halves matern
def matern5(x, y, rho): 
	d =np.linalg.norm(x-y)
	arg = np.sqrt(5)*(d/rho)
	val = (1 + arg + 1/3*arg**2)*np.exp(-arg) 
	return val

# 3 halves matern 
def matern3(x, y, rho): 
	d = np.linalg.norm(x-y) 
	arg = np.sqrt(3)*(d/rho) 
	return (1 + arg)*np.exp(-arg) 

def sqrexp(x, y, l): # Here x and y are vectors of stencil size
	arg = -(np.linalg.norm(x - y))**2 / (2.*l**2) 
	return np.exp(arg)

