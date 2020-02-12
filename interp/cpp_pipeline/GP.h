#ifndef INTERP_H
#define INTERP_H
#include <array>
#include <vector>
#include <cmath> 
#include "weights.h"
#include <iomanip>
#include <iostream>
class GP
{
public:

	/* Member data */ 
	std::vector<std::array<std::array<float, 9>, 9> > weight;
	std::vector<std::array<float, 9> > gammas;
	std::array<std::array<float, 9>, 9> vectors; 
	std::array<float, 9> eigen;
    std::array<float, 9> C; 
	int insize[3];

	/* Constructor */ 
	GP(std::vector<std::array<std::array<float, 9>, 9> > wts, std::vector<std::array<float, 9> > gam, 
	   std::array<std::array<float, 9>, 9> vec, std::array<float, 9> eig, const int size[3]){
		weight = wts;
		gammas = gam;
		vectors = vec; 
		eigen = eig; 
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	GP(const weights wgts, const int size[3]){
		weight = wgts.ks;
    	gammas = wgts.gam; 
		vectors = wgts.V;
		eigen = wgts.lam;
        C = wgts.C; 
		insize[0] = size[0], insize[1] = size[1], insize[2] = size[2]; 
	}

	/* Member functions */ 
	inline
	float dot(const std::array<float, 9> &vec1, const std::array<float, 9> &vec2){
        float result = 0; 
        #pragma unroll
        for(int i = 0; i< 9; i++) result += vec1[i]*vec2[i]; 
		return result; 
	}	

	inline
	std::array<float, 9> load(const std::vector<float> &img_in, 
		       	        	  const int j, const int i)
	{
		std::array<float, 9> result;
        int left = (i-1);
        int right = (i+1);
		int id0 =  (j-1)*insize[0]; 
		int id1 =  j*insize[0]; 
        int id2 =  (j+1)*insize[0];
		result[0] = img_in[id0 + left];
		result[1] = img_in[id1 + left];
		result[2] = img_in[id2 + left];
		result[3] = img_in[id0 + i]; 
		result[4] = img_in[id1 + i];
		result[5] = img_in[id2 + i]; 
		result[6] = img_in[id0 + right];
		result[7] = img_in[id1 + right]; 
		result[8] = img_in[id2 + right]; 
		return result; 	
	}
 
	inline
	std::array<float, 9> load_borders(const std::vector<float> &img_in, 
                		       		  const int j, const int i)
	{
		std::array<float, 9> result;
		int bot = (j-1 < 0) ? 0 : j-1;
		int left = (i-1 < 0) ? 0 : i-1; 
		int top = (j+1 >= insize[1]) ? insize[1]-1 : j+1; 
		int right = (i+1 >= insize[0]) ? insize[0]-1 : i+1;
		int id0 =  bot*insize[0]; 
		int id1 =  j*insize[0]; 
        int id2 =  top*insize[0];
		result[0] = img_in[id0 + left];
		result[1] = img_in[id1 + left];
		result[2] = img_in[id2 + left];
		result[3] = img_in[id0 + i]; 
		result[4] = img_in[id1 + i];
		result[5] = img_in[id2 + i]; 
		result[6] = img_in[id0 + right];
		result[7] = img_in[id1 + right]; 
		result[8] = img_in[id2 + right]; 
		return result; 	
	} 

	inline
	std::array<float, 9> load(const std::vector<float> &img_in, 
		       		  const int k, const int j, const int i)
	{
		std::array<float, 9> result;
		int bot = (j-1 < 0) ? 0 : j-1;
		int left = (i-1 < 0) ? 0 : i-1; 
		int top = (j+1 >= insize[1]) ? insize[1]-1 : j+1; 
		int right = (i+1 >= insize[0]) ? insize[0]-1 : i+1;
		int id0 = (k*insize[1] + bot)*insize[0]; 
		int id1 = (k*insize[1] + j)*insize[0]; 
        int id2 = (k*insize[1] + top)*insize[0];
		result[0] = img_in[id0 + left];
		result[1] = img_in[id1 + left];
		result[2] = img_in[id2 + left];
		result[3] = img_in[id0 + i]; 
		result[4] = img_in[id1 + i];
		result[5] = img_in[id2 + i]; 
		result[6] = img_in[id0 + right];
		result[7] = img_in[id1 + right]; 
		result[8] = img_in[id2 + right];
		return result; 	
	}

	inline
	void get_beta(const std::array<float, 9> &lbot, const std::array<float, 9> &bot,
                  const std::array<float, 9> &rbot, const std::array<float, 9> &left, 
                  const std::array<float, 9> &cen,  const std::array<float, 9> &right,
                  const std::array<float, 9> &ltop, const std::array<float, 9> &top, 
                  const std::array<float, 9> &rtop, std::array<float, 9> &beta)
	{
		// beta = f^T K^(-1) f = sum 1/lam *(V^T*f)^2 
		std::array<float, 9> vs = {}; 
/*    	if(!FLAG){
		    beta[4] = 0.f; 
                    for(int i =0; i < 9; ++i){
			vs = vectors[i]; 
			float prod = GP::dot(vs, cen); 
			beta[4] += (1.f*eigen[i])*(prod*prod); 
		    }
		} */ 
        for(int i =0; i < 9; ++i) beta[i] = 0.f; 
		for(int i =0; i < 9; ++i){
			vs = vectors[i];
			float prod = GP::dot(vs, lbot);
			beta[0] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, bot);
			beta[1] += (1.f/eigen[i])*(prod*prod);
 
			prod = GP::dot(vs, rbot);
			beta[2] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, left);
			beta[3] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, cen);
			beta[4] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, right);
			beta[5] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, ltop);
			beta[6] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, top);
			beta[7] += (1.f/eigen[i])*(prod*prod); 

			prod = GP::dot(vs, rtop);
			beta[8] += (1.f/eigen[i])*(prod*prod);
		}
	}

    inline float getalpha(const std::array<float, 9> &cen){
        float bets = 0.f;
        float avg = 0.f; 
        for(int i =0; i < 9; i++){
            float prod = GP::dot(vectors[i], cen);
            avg += cen[i];
            bets += (1.f/eigen[i])*(prod*prod); 
        }
        avg /= 9; 
       return bets/(avg*avg); 
    }

    inline
    std::array<float, 9> getMSweights(const std::array<float, 9> &beta, const int ksid){
        std::array<float, 9> w8ts; 
        float summ = 0; 
        for (int i = 0; i < 9; i++){
            float denom = std::pow((beta[i] + 1e-20), 2);
            w8ts[i] = gammas[ksid][i]/denom;
            summ += w8ts[i]; 	
        } 
        for(int i = 0; i < 9; i++) w8ts[i] /= summ;
        return w8ts; 
    }

	inline
	float combine(const std::array<float, 9> &lbot, const std::array<float, 9> &bot, const std::array<float, 9> &rbot,
		          const std::array<float, 9> &left, const std::array<float, 9> &cen, const std::array<float, 9> &right, 
		          const std::array<float, 9> &ltop, const std::array<float, 9> &top, const std::array<float, 9> &rtop, 
		          const std::array< std::array<float, 9>, 9> &wts, const std::array<float, 9> &wsm)
	{
		std::array<float, 9> gp; 
		gp[0] = GP::dot(wts[0], lbot); 
		gp[1] = GP::dot(wts[1], bot); 
		gp[2] = GP::dot(wts[2], rbot); 
		gp[3] = GP::dot(wts[3], left);
		gp[4] = GP::dot(wts[4], cen); 
		gp[5] = GP::dot(wts[5], right);
		gp[6] = GP::dot(wts[6], ltop); 
		gp[7] = GP::dot(wts[7], top); 
		gp[8] = GP::dot(wts[8], rtop); 
		float summ = 0;
		for(int i = 0; i < 9; i++){
            
			summ += wsm[i]*gp[i];
		}
		return summ; 
	}

	void single_channel_interp(const std::vector<float> img_in, 
		      std::vector<float> &img_out, const int ry, const int rx); 
	void single_channel_interp_base(const std::vector<float> img_in, 
		      std::vector<float> &img_out, const int ry, const int rx); 
    
	void MSinterp(const std::vector<float> img_in, 
		      std::vector<float> &img_out, const int ry, const int rx); 
};

#endif
