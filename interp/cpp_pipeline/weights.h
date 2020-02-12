#ifndef WEIGHTS_H
#define WEIGHTS_H
#include <vector>
#include <array> 
#include <cmath> 

// weights class for GP superresoltuion

class weights
{
    public: 
    weights(const int ratio[], const float del[]);
    ~weights(){};      
    
    // Member data
    int r[2]; 
    float dx[2];
    std::array<float, 9> C = {};   
    //
    // Eigen Values of Covariance Matrix
    //
    std::array<float, 9> lam = {}; 
    //
    //  Eigen Vectors of Covariance Matrix
    //
    std::array<std::array<float, 9>, 9> V = {};
    //
    //  Weights to be applied for interpolation
    //
    std::vector<std::array<std::array<float, 9>, 9> > ks; 
    //
    //  Gammas needed for optimal combination
    //
    std::vector<std::array<float, 9> > gam;
    float l;
    float sig;  

// Linear Algebra Functions
    template<int n>
    inline
    static float inner_prod(const float x[n], const float y[n])
    {
        float result = 0.f; 
        for(int i = 0; i < n; ++i) result += x[i]*y[i];
        return result;  
    }

    template<int n> 
    void
    cholesky(float (&b)[n], std::array<std::array<float, n>, n> K); 
    
    template<int n> 
    void
    cholesky(std::array<float, n> &b, std::array<std::array<float, n>, n> K);

    template<int n> 
    void
    cholesky(std::array<double, n> &b, std::array<std::array<double, n>, n> K);

    void
    Decomp(std::array<std::array<double, 9>, 9> &K, std::array<std::array<double, 25>, 25> &Kt); 

    // Set up for the multi-sampled Weighted GP interpolation 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    // And for Total Stencil
    void GetK(std::array<std::array<double, 9>, 9> &K, std::array<std::array<double, 25>, 25> &Ktot); 
    //
    // Get Weights builds k*Kinv for each stencil
    //
    void GetKs(const std::array<std::array<double, 9>, 9> &K, 
              std::vector<std::array<std::array<double,9>, 9> > &k);
    //
    //  Get Weights for the LSQ RHS
    //
    void GetKtotks(const std::array<std::array<double, 25>, 25> K1, std::vector<std::array<double, 25> > &kt); 
    //
    // Get Gamma by solving a LSQ problem only need this once. 
    //
    void GetGamma(std::array<std::array<double, 9>, 9> const& k,
                  std::array<double, 25> &kt, std::array<float, 9> &ga); 
    //
    //  Get EigenVecs and EigenValues for smoothness indicators. 
    //  Will use the Shifted QR algorithm with deflation 
    //
    void GetEigen();
};



#endif 
