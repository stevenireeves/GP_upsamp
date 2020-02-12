#ifndef GP_KERNELS_H
#define GP_KERNELS_H
#include <cmath>
#include <array>

template<class T> 
inline
T matern3(std::array<T, 2> x, std::array<T, 2> y, T rho, T dx[]){
	T arg = 0;
	T norm = 0;
	for(int i = 0; i < 2; ++i)
	       	norm += (x[i] - y[i])*dx[i]*(x[i] - y[i])*dx[i];
	arg = std::sqrt(3*norm)/rho;
	return (1 + arg)*std::exp(-arg);
//    arg = std::sqrt(norm)/rho; 
//    return (1. + std::sqrt(5.)*arg + 5./3.*arg*arg)*std::exp(-std::sqrt(5)*arg);
} // */

/*
template<class T> 
inline
T matern3(std::array<T, 2> x, std::array<T, 2> y, T rho, T dx[]){
	T arg = 0;
	T norm = 0;
	for(int i = 0; i < 2; ++i)
	       	norm += (x[i] - y[i])*dx[i]*(x[i] - y[i])*dx[i];
    arg = -0.5*(norm/(rho*rho)); 
    return std::exp(arg); 
}
// */
#endif 
