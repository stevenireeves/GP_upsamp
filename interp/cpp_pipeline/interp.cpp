/* This is the wrapper a c-style wrapper for for the GP interpolation to be used from python. */ 
#include "GP.h"
#include "weights.h"
#include <iostream>

#ifdef WITH_PY
void driver(const uint8_t img_in[], float img_out[], const int upsample_ratio[], const int in_size[], const float ks[], const float c_in[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    weights wgts(upsample_ratio, ks, c_in); 
    const int size[3] = {in_size[0], in_size[1], 1}; 
    GP interp(wgts, size); 
    interp.single_channel_interp(img_in, img_out, upsample_ratio[0], upsample_ratio[1]);
}

extern "C"
{
	void interpolate(const uint8_t *img_in, float *img_out, const int *upsample_ratio, const int *in_size, const float *ks, const float *cin){
			driver(img_in, img_out, upsample_ratio, in_size, ks, cin); 
	}
}

#else
void driver(const uint8_t *img_in, float *img_out, const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], 1}; 
    GP interp(wgts, size); 
    interp.single_channel_interp(img_in, img_out, upsample_ratio[0], upsample_ratio[1]);
}

extern "C"
{
	void interpolate(const uint8_t *img_in, float *img_out, const int *upsample_ratio, const int *in_size){
			driver(img_in, img_out, upsample_ratio, in_size); 
	}

}
#endif 
