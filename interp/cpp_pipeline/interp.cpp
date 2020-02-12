/* This is the wrapper a c-style wrapper for for the GP interpolation to be used from python. */ 
#include "GP5.h"
#include "weights5.h"
#include <iostream>

void driver(float *img_in, float *img_out, const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    std::vector<float> img1(img_in, img_in + in_size[0]*in_size[1]); 
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], 1}; 
    GP interp(wgts, size); 
    std::vector<float> img2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    interp.single_channel_interp_base(img1, img2, upsample_ratio[0], upsample_ratio[1]);
    std::copy(img2.begin(), img2.end(), img_out);
}

void driver_color(float *bin, float *gin, float *rin, 
                  float *bout, float *gout, float *rout,
                  const int upsample_ratio[], const int in_size[]){
    const float del[2] = {1.f/float(in_size[0]), 1.f/float(in_size[1])};
    std::vector<float> b1(bin, bin + in_size[0]*in_size[1]);
    std::vector<float> g1(gin, gin + in_size[0]*in_size[1]); 
    std::vector<float> r1(rin, rin + in_size[0]*in_size[1]);  
    weights wgts(upsample_ratio, del); 
    const int size[3] = {in_size[0], in_size[1], in_size[2]}; 
    GP interp(wgts, size); 
    std::vector<float> b2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    std::vector<float> r2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    std::vector<float> g2(size[0]*upsample_ratio[0]*upsample_ratio[1]*size[1], 0.f);
    interp.single_channel_interp_base(b1, b2, upsample_ratio[0], upsample_ratio[1]);
    interp.single_channel_interp_base(g1, g2, upsample_ratio[0], upsample_ratio[1]);
    interp.single_channel_interp_base(r1, r2, upsample_ratio[0], upsample_ratio[1]);
    std::copy(b2.begin(), b2.end(), bout); 
    std::copy(g2.begin(), g2.end(), gout); 
    std::copy(r2.begin(), r2.end(), rout); 
}

extern "C"
{
	void interpolate(float *img_in, float *img_out, const int *upsample_ratio, const int *in_size){
			driver(img_in, img_out, upsample_ratio, in_size); 
	}

    void interpolate_color(float *b_in, float *g_in, float* r_in, 
                           float *b_out, float *g_out, float *r_out, 
                           const int *upsample_ratio, const int *in_size){
			driver_color(b_in, g_in, r_in,
                         b_out, g_out, r_out, 
                         upsample_ratio, in_size);
	}
}


