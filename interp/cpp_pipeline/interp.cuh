#include <cuda_runtime.h>
#include <cmath>
#include "img.cuh"

__global__
void gray_interp_cu(const img img_in, img img_out, const int ry, const int rx)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y;
    if(i < img.width){
	    if(j < img.height){
		    float beta[9];
		    float cen[9];
		    GP::load(img_in,cen, j, i);
		    float alpha = GP::getalpha(cen, beta[4]);
		    if(alpha > 50){ 
			float lbot[9]; 
			float bot[9];
			float rbot[9]; 
			float left[9];
			float right[9];
			float ltop[9];
			float top[9]; 
			float rtop[9];
			GP::load(img_in, lbot , j-1, i-1);
			GP::load(img_in, bot  , j  , i-1);
			GP::load(img_in, rbot , j+1, i-1);
			GP::load(img_in, left , j-1, i  );
			GP::load(img_in, right, j+1, i  );
			GP::load(img_in, ltop , j-1, i+1);
			GP::load(img_in, top  , j  , i+1);
			GP::load(img_in, rtop , j+1, i+1);
			GP::get_beta(lbot, bot, rbot , 
				     left, cen, right,
				     ltop, top, rtop , beta, true);

			for(int idy = 0; idy < ry; idy++){
			    int jj = j*ry + idy;
			    int ind =jj*outsize[1];	
			    for(int idx = 0; idx < rx; idx++){
				int ii = i*rx + idx; 
				int idk = idx*ry + idy;
				auto msweights = GP::getMSweights(beta, idk);
				img_out[ind + ii] = GP::combine(lbot, bot, rbot ,
								left, cen, right,
								ltop, top, rtop , 
								weight[idk], msweights);
			    }
			}
		    }
		    else{
			for(int idy = 0; idy < ry; idy++){
			    int jj = j*ry + idy;
			    int ind =jj*outsize[1];	
			    for(int idx = 0; idx < rx; idx++){
				int ii = i*rx + idx; 
				int idk = idx*ry + idy;
				img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
			    }
			}
		    }
		}
	}
}


__global__ void interp(unsigned char * d_src, unsigned char * d_dst, int width, int height)
{
    int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pos_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (pos_x >= width || pos_y >= height)
        return;
	
    uchar3 rgb;
    rgb.x = d_src[pos_y * width + pos_x];
    rgb.y = d_src[(height + pos_y ) * width + pos_x];
    rgb.z = d_src[(height * 2 + pos_y) * width + pos_x];

    unsigned int _gray = (unsigned int)(0.299f*rgb.x + 0.587f*rgb.y + 0.114*rgb.z);
    unsigned char gray = _gray > 255 ? 255 : _gray;

    d_dst[pos_y * width + pos_x] = gray;
}


template<size_t n> 
__device__ inline float dot(float vec1[n], float vec2[n]){
	float summ = 0.f; 
#pragma unroll
	for(int i = 0; i < n; i++) summ += vec1[i]*vec2[i]; 
	return summ; 

}






