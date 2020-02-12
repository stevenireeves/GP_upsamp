/* This is the wrapper a c-style wrapper for for the GP interpolation to be used from python. */ 
#include "GP.h"
#include "weights.h"
#include <iostream>

void driver(float *img_in, float *img_out, const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
//    std::vector<float> img1(img_in, img_in + in_size[0]*in_size[1]); 
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], 1}; 
    GP interp(wgts, size); 
//    std::vector<float> img2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    interp.single_channel_interp(img_in, img_out, upsample_ratio[0], upsample_ratio[1]);
//    std::copy(img2.begin(), img2.end(), img_out);
}

extern "C"
{
	void interpolate(float *img_in, float *img_out, const int *upsample_ratio, const int *in_size){
			driver(img_in, img_out, upsample_ratio, in_size); 
	}

}


