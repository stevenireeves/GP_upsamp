#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <vector>
#include <array> 
#include <cmath> 

// weights class for GP 25 superresoltuion

class weights
{
    public: 
    weights(const int ratio[], const float del[]);
    weights(const int ratio[], const float ks_in[], const float C_in[]); 
    ~weights(){};      
    
    // Member data
    int r[2]; 
    float dx[2];
    std::array<float, 25> C = {};   
    //
    //  Weights to be applied for interpolation
    //
    std::vector<std::array<float, 25> > ks; 
    float l;
   

// Linear Algebra Functions
    template<int n> 
    void
    cholesky(std::array<double, n> &b, std::array<std::array<double, n>, n> K);

    void
    Decomp(std::array<std::array<double, 25>, 25> &K); 

    // Set up for the multi-sampled Weighted GP interpolation 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    // And for Total Stencil
    void GetK(std::array<std::array<double, 25>, 25> &K); 
    //
    // Get Weights builds k*Kinv for each stencil
    //
    void Getks(const std::array<std::array<double, 25>, 25> K, std::vector<std::array<double, 25> > &k); 
};



#endif 
