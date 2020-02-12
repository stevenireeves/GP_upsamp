#include <iostream>
#include <omp.h>
#include "GP.h" 

void GP::single_channel_interp_base(const std::vector<float> img_in,
			  std::vector<float> &img_out, const int ry, const int rx)
{
	const int outsize[2] = {insize[1]*ry, insize[0]*rx}; 
	/*------------------- Body -------------------------- */
#pragma omp parallel for	
	for( int j = 1; j < insize[1]-1; j++){
		for(int i = 1; i < insize[0]-1; i++){
            auto cen = GP::load(img_in, j, i);
		    for(int idy = 0; idy < ry; idy++){
		        int jj = j*ry + idy;
		        int ind =jj*outsize[1];	
		        for(int idx = 0; idx < rx; idx++){
			        int ii = i*rx + idx; 
			        int idk = idx*ry + idy;
                    float mle = GP::dot(C, cen); 
                    std::array<float, 9> f; 
                    for(int kk = 0; kk < 9; kk++) f[kk] = cen[kk] - mle; 
			        img_out[ind + ii] = mle +  GP::dot(weight[idk][4], f); 
		            }
	            }
	      }
	}
	/*---------------- Borders ------------------------- */
	//================ bottom =========================
	int j = 0; 
#pragma omp parallel for
	for(int i = 0; i < insize[0]; i++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int ind = idy*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx; 
				int idk = idx*ry + idy;
                float mle = GP::dot(C, cen); 
                std::array<float, 9> f; 
                for(int kk = 0; kk < 9; kk++) f[kk] = cen[kk] - mle; 
                img_out[ind + ii] = mle +  GP::dot(weight[idk][4], f); 
			}
		}
	}
	//================= top =======================
	j = (insize[1]-1); 	
#pragma omp parallel for
	for(int i = 0; i < insize[0]; i++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 	
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx;
				int idk = idx*ry + idy;
                float mle = GP::dot(C, cen); 
                std::array<float, 9> f; 
                for(int kk = 0; kk < 9; kk++) f[kk] = cen[kk] - mle; 
                img_out[ind + ii] = mle +  GP::dot(weight[idk][4], f); 
			}
		}
	}
	//============== left =========================
	int i = 0; 
#pragma omp parallel for
	for(j = 0; j < insize[1]; j++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind =jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int idk = idx*ry + idy;
                float mle = GP::dot(C, cen); 
                std::array<float, 9> f; 
                for(int kk = 0; kk < 9; kk++) f[kk] = cen[kk] - mle; 
                img_out[ind + idx] = mle +  GP::dot(weight[idk][4], f); 
			}
		}
	}
	//=============== right ======================
	i = (insize[0]-1); 
#pragma omp parallel for
	for(j = 0; j < insize[1]; j++){
		auto cen = GP::load_borders(img_in, j, i); 
		for(int idy = 0; idy < ry; idy++){
			int jj = idy + j*ry; 
			int ind = jj*outsize[1];
			for(int idx = 0; idx < rx; idx++){
				int ii = idx + i*rx; 
				int idk = idx*ry + idy;
                    float mle = GP::dot(C, cen); 
                    std::array<float, 9> f; 
                    for(int kk = 0; kk < 9; kk++) f[kk] = cen[kk] - mle; 
			        img_out[ind + ii] = mle +  GP::dot(weight[idk][4], f); 
			}
		}
	}
}

/* GP Interpolation with Single Gray Channel */ 
void GP::single_channel_interp(const std::vector<float> img_in, 
                     std::vector<float> &img_out, const int ry, const int rx){
	const int outsize[2] = {insize[1]*ry, insize[0]*rx}; 
	/*------------------- Body -------------------------- */
#pragma omp parallel for	
	for( int j = 2; j < insize[1]-2; j++){
		std::array<float, 9> beta = {};
		std::array<float, 9> leftbot = {}; 
		std::array<float, 9> bot = {};
		std::array<float, 9> rightbot = {}; 
		std::array<float, 9> left = {};
		std::array<float, 9> cen = {};
		std::array<float, 9> right = {};
		std::array<float, 9> lefttop = {};
		std::array<float, 9> top = {}; 
		std::array<float, 9> righttop = {};
		for(int i = 2; i < insize[0]-2; i++){
		    cen = GP::load(img_in, j, i);
		    float alpha = GP::getalpha(cen);
		    if(alpha > 0){
                leftbot = GP::load(img_in, j-1, i-1);
                bot = GP::load(img_in, j, i-1);
                rightbot = GP::load(img_in, j+1,i-1);
                left = GP::load(img_in, j-1, i);
                right = GP::load(img_in, j+1, i);
                lefttop = GP::load(img_in, j-1, i+1);
                top = GP::load(img_in, j, i+1);
                righttop = GP::load(img_in, j+1, i+1);
                GP::get_beta(leftbot, bot, rightbot, 
                         left   , cen, right   ,
                         lefttop, top, righttop, beta);
                for(int idy = 0; idy < ry; idy++){
                    int jj = j*ry + idy;
                    int ind =jj*outsize[1];	
                    for(int idx = 0; idx < rx; idx++){
                        int ii = i*rx + idx; 
                        int idk = idx*ry + idy;
                        auto msweights = GP::getMSweights(beta, idk);

                        img_out[ind + ii] = GP::combine(leftbot, bot, rightbot ,
                                        left   , cen, right    ,
                                        lefttop, top, righttop , 
                                        weight[idk], msweights );
                        
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
	/*---------------- Borders ------------------------- */
	//================ bottom =========================
	for(int j = 0; j <=1; j++){
#pragma omp parallel for
        for(int i = 0; i < insize[0]; i++){
            auto cen = GP::load_borders(img_in, j, i); 
            for(int idy = 0; idy < ry; idy++){
                int jj = j*ry + idy; 
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
                }
            }
        }
    }
	//================= top =======================
	for(int j = (insize[1]-2); j < insize[1]; j++){ 	
#pragma omp parallel for
        for(int i = 0; i < insize[0]; i++){
            auto cen = GP::load_borders(img_in, j, i); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 	
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx;
                    int idk = idx*ry + idy;
                    img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
                }
            }
        }
    }
	//============== left =========================
	for(int i = 0; i < 2; i++){ 
#pragma omp parallel for
        for(int j = 0; j < insize[1]; j++){
            auto cen = GP::load_borders(img_in, j, i); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 
                int ind =jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
                }
            }
        }
    }
	//=============== right ======================
	for(int i = (insize[0]-2); i < insize[0]; i++){ 
#pragma omp parallel for
        for(int j = 0; j < insize[1]; j++){
            auto cen = GP::load_borders(img_in, j, i); 
            for(int idy = 0; idy < ry; idy++){
                int jj = idy + j*ry; 
                int ind = jj*outsize[1];
                for(int idx = 0; idx < rx; idx++){
                    int ii = idx + i*rx; 
                    int idk = idx*ry + idy;
                    img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
                }
            }
        }
    }
}


/* GP interpolation with 3 Channels */ 
void GP::MSinterp(const std::vector<float> img_in, 
                        std::vector<float> &img_out, const int ry, const int rx){
	const int outsize[3] = {insize[0]*rx, insize[1]*ry, insize[0]*rx}; 
       	for( int k = 0; k < insize[2]; k++){
		/*------------------- Body -------------------------- */
#pragma omp parallel for	
		for( int j = 1; j < insize[1]-1; j++){
			std::array<float, 9> beta = {};
			std::array<float, 9> leftbot = {}; 
			std::array<float, 9> bot = {};
			std::array<float, 9> rightbot = {}; 
			std::array<float, 9> left = {};
			std::array<float, 9> cen = {};
			std::array<float, 9> right = {};
			std::array<float, 9> lefttop = {};
			std::array<float, 9> top = {}; 
			std::array<float, 9> righttop = {};
			for(int i = 1; i < insize[0]-1; i++){
				leftbot = GP::load(img_in, k, j-1, i-1);
				bot = GP::load(img_in, k, j, i-1);
				rightbot = GP::load(img_in, k, j+1,i-1);
				left = GP::load(img_in, k, j-1, i);
				cen = GP::load(img_in, k, j, i);
				right = GP::load(img_in, k, j+1, i);
				lefttop = GP::load(img_in, k, j-1, i+1);
				top = GP::load(img_in, k, j, i+1);
				righttop = GP::load(img_in, k, j+1, i+1);
				GP::get_beta(leftbot, bot, rightbot, 
				             left   , cen, right   ,
				  	     lefttop, top, righttop, beta);

				for(int idy = 0; idy < ry; idy++){
					int jj = j*ry + idy;
					int ind = (k*outsize[2] + jj)*outsize[0];	
					for(int idx = 0; idx < rx; idx++){
						int ii = i*rx + idx; 
						int idk = idx*ry + idy;
						auto msweights = GP::getMSweights(beta, idk);
						img_out[ind + ii] = GP::combine(leftbot, bot, rightbot ,
									        left   , cen, right    ,
								     	     	lefttop, top, righttop , 
									     	weight[idk], msweights);
					}
				}
			}
		}
		/*---------------- Borders ------------------------- */
	       	//================ bottom =========================
		int j = 0; 
#pragma omp parallel for
		for(int i = 0; i < insize[0]; i++){
			auto cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				int ind = (k*outsize[2] + idy)*outsize[0];
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx; 
					int idk = idx*ry + idy;
					img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//================= top =======================
		j = (insize[1]-1); 	
#pragma omp parallel for
		for(int i = 0; i < insize[0]; i++){
			auto cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				int jj = idy + j*ry; 	
				int ind = (k*outsize[2] + jj)*outsize[0];
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx;
					int idk = idx*ry + idy;
					img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//============== left =========================
		int i = 0; 
#pragma omp parallel for
		for(j = 0; j < insize[1]; j++){
			auto cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				int jj = idy + j*ry; 
				int ind = (k*outsize[2] + jj)*outsize[0];
				for(int idx = 0; idx < rx; idx++){
					int idk = idx*ry + idy;
					img_out[ind + idx] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
		//=============== right ======================
		i = (insize[0]-1); 
#pragma omp parallel for
		for(j = 0; j < insize[1]; j++){
			auto cen = GP::load(img_in, k, j, i); 
			for(int idy = 0; idy < ry; idy++){
				int jj = idy + j*ry; 
				int ind = (k*outsize[2] + jj)*outsize[0];
				for(int idx = 0; idx < rx; idx++){
					int ii = idx + i*rx; 
					int idk = idx*ry + idy;
					img_out[ind + ii] = GP::dot(weight[idk][4], cen); 
				}
			}
		}
	}
}

