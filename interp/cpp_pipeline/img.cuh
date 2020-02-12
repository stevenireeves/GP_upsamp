#ifndef IMG_H
#define IMG_H

class img
{
	public:
		/* Member Data */
		int width; 
		int height; 
		int size; 
		float *data; 

		/* Member Functions */ 
		img(const int w, const int h){
			width = w; 
			height = h; 
			size = w*h;
			cudaMalloc((void**)data,size*sizeof(float)); 
		}

		img(const int w, const int h, float dat[]){
			width = w; 
			height = h; 
			size = w*h;
			cudaMalloc(&data,size*sizeof(float)); 
			cudaMemcpy(data, dat, size*sizeof(float), cudaMemcpyHostToDevice); 		
		}

		void get_data_host(float data_h[]){
			cudaMemcpy(data_h,data, size*sizeof(float), cudaMemcpyHostToDevice); 
		}

		void deallocate(){ /* Note this must be called before the destructor */ 
			cudaFree(data);
		}
		__device__ &operator[] (int index){
			return data[index];
		}
}

#endif
