#include <iostream>
#include <omp.h>
#include "GP5.h"
 
/* GP Interpolation with Single Channel */ 
void GP::single_channel_interp_base(const std::vector<float> img_in, 
                                    std::vector<float> &img_out, const int ry, const int rx){
	const int outsize[2] = {insize[1]*ry, insize[0]*rx}; 
	/*------------------- Body -------------------------- */
#pragma omp parallel for	
	for( int j = 2; j < insize[1]-2; j++){
		std::array<float, 25> cen = {};
		for(int i = 2; i < insize[0]-2; i++){
		    cen = GP::load(img_in, j, i);
            float mle = GP::dot(C, cen); 
            for(int idy = 0; idy < ry; idy++){
                int jj = j*ry + idy;
                int ind =jj*outsize[1];	
                for(int idx = 0; idx < rx; idx++){
                    int ii = i*rx + idx; 
                    int idk = idx*ry + idy;
                    std::array<float, 25> f; 
                    for(int kk = 0; kk < 25; kk++) f[kk] = cen[kk] - mle; 
                    img_out[ind + ii] = mle + GP::dot(weight[idk], f); 
                }
            }
       	}
	}
	/*---------------- Borders ------------------------- */
	//================ bottom =========================
	for(int j = 0; j <=1; j++){
#pragma omp parallel for
        for(int i = 0; i < insize[0]; i++){
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            for(int idy = 0; idy < ry; idy++){
                int jj = j*ry + idy; 
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    std::array<float, 25> f; 
                    for(int kk = 0; kk < 25; kk++) f[kk] = cen[kk] - mle; 
                    img_out[ind + ii] = mle + GP::dot(weight[idk], f); 
                }
            }
        }
    }
	//================= top =======================
	for(int j = (insize[1]-2); j < insize[1]; j++){ 	
#pragma omp parallel for
        for(int i = 0; i < insize[0]; i++){
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 	
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx;
                    int idk = idx*ry + idy;
                    std::array<float, 25> f; 
                    for(int kk = 0; kk < 25; kk++) f[kk] = cen[kk] - mle; 
                    img_out[ind + ii] = mle + GP::dot(weight[idk], f); 
                }
            }
        }
    }
	//============== left =========================
	for(int i = 0; i < 2; i++){ 
#pragma omp parallel for
        for(int j = 0; j < insize[1]; j++){
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 
                int ind =jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    std::array<float, 25> f; 
                    for(int kk = 0; kk < 25; kk++) f[kk] = cen[kk] - mle; 
                    img_out[ind + ii] = mle + GP::dot(weight[idk], f); 
                }
            }
        }
    }
	//=============== right ======================
	for(int i = (insize[0]-2); i < insize[0]; i++){ 
#pragma omp parallel for
        for(int j = 0; j < insize[1]; j++){
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    std::array<float, 25> f; 
                    for(int kk = 0; kk < 25; kk++) f[kk] = cen[kk] - mle; 
                    img_out[ind + ii] = mle + GP::dot(weight[idk], f); 
                }
            }
        }
    }
}


