# Gaussian Process Upsampling method for Grayscale Document Images.

This is the implementation for the method outlined in the paper "A Gaussian Process Based Alogirthm 
to Upsample Document Images for Optical Character Recognition" by Steven I Reeves et al. https://arxiv.org/pdf/2005.03780v1.pdf

What is included is a C++ library written to be a shared object for Python or called directly from C or C++.
Currently only 4x and 2x upsampling (per dim) is supported for grayscale images or single channel arrays.

The make system builds on Linux and MacOS, and requires "Lapacke", the C++ extension of the Linear Algebra Package LAPACK.
You may need to alter the Makefile so that the linker will find this package. 

Additionally, there is the option to use multi-threading in this application. To generate multi-threaded capable 
library, type: 

make USE_OMP=T -j2


If the application needed requires minimal C++ library dependence, build the C++ library with 
make WITH_PY=T USE_OMP=T -j2
and use the functions in py_pipeline to generate the GP model weights. The script gptest.py contains this usage.  
