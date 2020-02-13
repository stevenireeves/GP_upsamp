# GP_upsamp
Gaussian Process Upsampling method for Grayscale Images. 
This is the implementation for the method outlined in the paper "A Gaussian Process Based Alogirthm 
to Upsample Document Images for Optical Character Recognition" by Steven I Reeves et al. 

What is included is a C++ library written to be a shared object for Python or called directly from C or C++.
Currently only 4x and 2x upsampling (per dim) is supported for grayscale images or single channel arrays.

The make system builds on Linux and MacOS, and requires "Lapacke", the C++ extension of the Linear Algebra Package LAPACK.
You may need to alter the Makefile so that the linker will find this package. 
