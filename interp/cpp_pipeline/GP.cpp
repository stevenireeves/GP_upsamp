#include <iostream>
#include <stdint.h>
#include <omp.h>
#include "GP.h"
 
/* GP Interpolation with Single Channel */ 
void GP::single_channel_interp(const uint8_t* img_in, 
                               float* img_out, const int ry, const int rx){
	const int outi = insize[1]*rx; 
	/*------------------- Body -------------------------- */
#pragma omp parallel for	
	for(int i = 2; i < insize[0]-2; i++){
		for( int j = 2; j < insize[1]-2; j++){
		    auto cen = GP::load(img_in, j, i);
            float mle = GP::dot(C, cen); 
            GP::sub(cen, mle); 
            for(int idx = 0; idx < rx; idx++){
                int ii = i*rx + idx;
                int ind =ii*outi;
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy; 
                    int idk = idx*ry + idy;
                    img_out[ind + jj] = mle + GP::dot(weight[idk], cen); 
                }
            }
       	}
	}
	/*---------------- Borders ------------------------- */
	//================ top =========================
#pragma omp parallel for
    for(int i = 0; i < insize[0]; i++){
	    for(int j = 0; j < 2; j++){
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            GP::sub(cen, mle); 
            for(int idx = 0; idx < rx; idx++){
                int ii = i*rx + idx;
                int ind =ii*outi;
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy; 
                    int idk = idx*ry + idy;
                    img_out[ind + jj] = mle + GP::dot(weight[idk],cen); 
                }
            }
        }
    }
	//================= bottom =======================
#pragma omp parallel for
    for(int i = 0; i < insize[0]; i++){
    	for(int j = (insize[1]-2); j < insize[1]; j++){ 	
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            GP::sub(cen, mle); 
            for(int idx = 0; idx < rx; idx++){
                int ii = i*rx + idx;
                int ind =ii*outi;
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy; 
                    int idk = idx*ry + idy;
                    img_out[ind + jj] = mle + GP::dot(weight[idk], cen); 
                }
            }
        }
    }
	//============== left =========================
#pragma omp parallel for
    for(int j = 0; j < insize[1]; j++){
    	for(int i = 0; i < 2; i++){ 
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            GP::sub(cen, mle); 
            for(int idx = 0; idx < rx; idx++){
                int ii = i*rx + idx;
                int ind =ii*outi;
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy; 
                    int idk = idx*ry + idy;
                    img_out[ind + jj] = mle + GP::dot(weight[idk], cen); 
                }
            }
        }
    }
	//=============== right ======================
#pragma omp parallel for
    for(int j = 0; j < insize[1]; j++){
    	for(int i = (insize[0]-2); i < insize[0]; i++){ 
            auto cen = GP::load_borders(img_in, j, i); 
            float mle = GP::dot(C, cen); 
            GP::sub(cen, mle); 
            for(int idx = 0; idx < rx; idx++){
                int ii = i*rx + idx;
                int ind =ii*outi;
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy; 
                    int idk = idx*ry + idy;
                    img_out[ind + jj] = mle + GP::dot(weight[idk], cen); 
                }
            }
        }
    }
}


