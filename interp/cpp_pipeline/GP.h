#ifndef INTERP_H
#define INTERP_H
#include <array>
#include <vector>
#include <cmath> 
#include <stdint.h>
#include "weights.h"
class GP
{
public:

	/* Member data */ 
	std::vector<std::array<float, 25> > weight;
    std::array<float, 25> C; 
	int insize[3];

	GP(const weights wgts, const int size[3]){
		weight = wgts.ks;
    	C = wgts.C; 
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	/* Member functions */ 
	inline
	float dot(const std::array<float, 25> &vec1, const std::array<float, 25> &vec2){
        float result = 0; 
        for(int i = 0; i< 25; i++) result += vec1[i]*vec2[i]; 
		return result; 
	}
    inline
    void sub(std::array<float, 25> &samp, const float mle)
    {
        for(int i = 0; i < 25; i++) samp[i] -= mle; 
    }

	inline
	std::array<float, 25> load(const uint8_t* img_in, 
		       	        	  const int j, const int i)
	{
		std::array<float, 25> result;
        int idi[5] = {(i-2)*insize[1], (i-1)*insize[1], i*insize[1], (i+1)*insize[1], (i+2)*insize[1]}; 
        int idj[5] = { j-2, j-1, j, j+1, j+2 }; 

        for(int i1 = 0; i1 < 5; i1++){
            for(int j1 = 0; j1 < 5; j1++){
                result[i1*5 + j1] = float(img_in[idi[i1] + idj[j1]]);  
            }
        }
		return result; 	
	}
 
	inline
	std::array<float, 25> load_borders(const uint8_t* img_in, 
                		       		  const int j, const int i)
	{
		std::array<float, 25> result;
        int idj[5]; 
        int idi[5]; 
        int k = 0; 
        for(int i1 = j-2; i1 <= j+2; i1++){
            if(i1 < 0) idj[k] = 0; 
            else if(i1 >= insize[1]) idj[k] = insize[1]-1; 
            else idj[k] = i1;
            k++;
        }

        k = 0; 
        for( int i1 = i-2; i1 <= i+2; i1++){
            if(i1 < 0) idi[k] = 0; 
            else if(i1 >= insize[0]) idi[k] = insize[0]-1; 
            else idi[k] = i1;
            k++;
        }
        for(int i1 = 0; i1 < 5; i1++){
            for(int j1 = 0; j1 < 5; j1++){ 
                result[i1*5 + j1] = float(img_in[(idi[i1]*insize[1] + idj[j1])]);  
            }
        }
		return result; 	
	} 

	void single_channel_interp(const uint8_t* img_in, 
		                      float* img_out, const int ry, const int rx); 
    
};

#endif
