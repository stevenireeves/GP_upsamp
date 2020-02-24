#include "weights.h"
#include "kernels.h"
#include <iostream>
#ifndef WITH_PY
    #include <lapacke.h> 
#endif


//Constructor 
#ifdef WITH_PY
weights::weights (const int Ratio[], const float ks_in[], const float C_in[])
{
    int factor = Ratio[0]*Ratio[1]; 
    for(int i = 0; i < 25; i++){
        C[i] = C_in[i]; 
        for(int j = 0; j < factor; j++){
            ks[j][i] = ks_in[i*factor + j]; //Need to check ordering
        }
    }
}

#else
weights::weights (const int Ratio[], const float del[])
{
    dx[0] = del[0], dx[1] = del[1]; 
    r[0] = Ratio[0], r[1] = Ratio[1];
    float delta = std::min(dx[0], dx[1]); 
    l = 20*delta; 
    std::array<std::array<double, 25>, 25> K = {}; // The same for every ratio; 
    std::vector<std::array<double, 25> > kis(r[0]*r[1], std::array<double, 25>{{0}});
    ks.resize(r[0]*r[1], std::array<float, 25>() );
         // First dim is rx*ry; 
    GetK(K); // Builds Covariance Matrices of Base Sample and Extended Samples/stencils  
    Decomp(K); //Decomposes K and Ktot into their Cholesky Versions
    //Build weights for Maximum Likelihood Estimate. 
    std::array<double, 25> ones;
    for(int i = 0; i < 25; i++) ones[i] = 1.;  
    cholesky<25>(ones, K); 
    double sum = 0.; 
    for(int i = 0; i < 25; i++){
        sum += ones[i]; 
    }
    for(int i = 0; i < 25; i++){
         C[i] = float(ones[i]/sum);
    } 
    //Get interpolation weights.
    Getks(K, kis);
}

template<int n> 
void
weights::cholesky(std::array<double, n> &b, std::array<std::array<double, n>, n> K)
{
    /* Forward sub Ly = b */
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j) b[i] -= b[j]*K[i][j];
        b[i] /= K[i][i];
    }

    /* Back sub Ux = y */
    for(int i = n-1; i >= 0; --i){
        for(int j = i+1; j < n; ++j) b[i] -= K[j][i]*b[j];
        b[i] /= K[i][i];
    }
}


void
weights::GetK(std::array<std::array<double, 25>, 25> &Ktot)
{
    std::array<std::array<double, 2>, 25>  spnt = {{{-2, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-2,  2}, 
                                                    {-1, -2}, {-1, -1}, {-1, 0}, {-1,  1}, {-1,  2},
                                                    { 0, -2}, { 0, -1}, { 0, 0}, { 0,  1}, { 0,  2}, 
                                                    { 1, -2}, { 1, -1}, { 1, 0}, { 1,  1}, { 1,  2}, 
                                                    { 2, -2}, { 2, -1}, { 2, 0}, { 2,  1}, { 2,  2}}}; 
    double del[2] = {double(dx[0]), double(dx[1])}; 
    for(int i = 0; i < 25; ++i)
        for(int j = i; j <25; ++j){
            Ktot[i][j] = matern3(spnt[i], spnt[j], double(l), del); 
            Ktot[j][i] = Ktot[i][j]; 
        }
}

void
weights::Decomp(std::array<std::array<double, 25>, 25>  &Kt)
{
    std::vector<double> temp(625, 0); 
    for(int i = 0; i < 25; i++)
        for(int j = 0; j < 25; j++)
            temp[i*25 + j] = Kt[i][j]; 
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', 25, temp.data(), 25); 
     for(int i = 0; i < 25; i++)
        for(int j = i; j < 25; j++){
            Kt[i][j] = temp[i*25+j];
            Kt[j][i] = Kt[i][j]; 
        } 
}



// Here we resolve the model weights for each newly interpolate point. 
// In this case, we will have 16 new points
// Therefore, we will need 16 b =  k*^T Ktot^(-1)
// K1 is already Choleskied  
void 
weights::Getks(const std::array<std::array<double, 25>, 25> K1, std::vector<std::array<double, 25> > &kt)
{
   //Locations of new points relative to i,j 
    std::vector<std::array<double,2>> pnt(r[0]*r[1], std::array<double,2>()); 
    if(r[0] == 2 && r[1] == 2){
        pnt[0][0] = -0.25,  pnt[0][1] = -0.25; 
        pnt[1][0] = -0.25,  pnt[1][1] =  0.25; 
        pnt[2][0] =  0.25,  pnt[2][1] = -0.25; 
        pnt[3][0] =  0.25,  pnt[3][1] =  0.25; 
    }
    else if(r[0] == 4 && r[1]==4){
        pnt[0][0]  = -.375, pnt[0][1]  = -.375; 
        pnt[1][0]  = -.375, pnt[1][1]  = -.125; 
        pnt[2][0]  = -.375, pnt[2][1]  = 0.125; 
        pnt[3][0]  = -.375, pnt[3][1]  = 0.375; 
        pnt[4][0]  = -.125, pnt[4][1]  = -.375; 
        pnt[5][0]  = -.125, pnt[5][1]  = -.125; 
        pnt[6][0]  = -.125, pnt[6][1]  = 0.125; 
        pnt[7][0]  = -.125, pnt[7][1]  = 0.375; 
        pnt[8][0]  = 0.125, pnt[8][1]  = -.375; 
        pnt[9][0]  = 0.125, pnt[9][1]  = -.125; 
        pnt[10][0] = 0.125, pnt[10][1] = 0.125; 
        pnt[11][0] = 0.125, pnt[11][1] = 0.375; 
        pnt[12][0] = 0.375, pnt[12][1] = -.375; 
        pnt[13][0] = 0.375, pnt[13][1] = -.125; 
        pnt[14][0] = 0.375, pnt[14][1] = 0.125; 
        pnt[15][0] = 0.375, pnt[15][1] = 0.375; 
    }

//  K points 
    std::array<std::array<double, 2>, 25>  spnt = {{{-2, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-2,  2}, 
                                                    {-1, -2}, {-1, -1}, {-1, 0}, {-1,  1}, {-1,  2},
                                                    { 0, -2}, { 0, -1}, { 0, 0}, { 0,  1}, { 0,  2}, 
                                                    { 1, -2}, { 1, -1}, { 1, 0}, { 1,  1}, { 1,  2}, 
                                                    { 2, -2}, { 2, -1}, { 2, 0}, { 2,  1}, { 2,  2}}}; 

    double del[2] = {double(dx[0]), double(dx[1])};
    for(int i = 0; i < r[0]*r[1]; i++){
       for (int j = 0; j < 25; j++){
            kt[i][j] = matern3(pnt[i], spnt[j], double(l), del); 
       }
       cholesky<25>(kt[i], K1);
       for(int j =0; j < 25; j++) ks[i][j] = float(kt[i][j]);  
    } 
}
#endif
