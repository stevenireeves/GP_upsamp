#include <stdio.h>
#include <iostream>

extern "C"
{
	void interpolate(float *img_in, float *img_out, const int *upsample_ratio, const int *in_size); 
}

int main(){
	float imgin[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; 
	float imgout[256] = {}; 
	int size[3] = {4, 4, 1};
	int upsample[2] = {4, 4}; 
	std::cout<<"Begin interpolate" << std::endl;
	interpolate(imgin, imgout, upsample, size); 
	for(int i = 0; i < 256; i++) std::cout << imgout[i] << std::endl;

	std::cout<<"End interpolate" << std::endl; 
}
