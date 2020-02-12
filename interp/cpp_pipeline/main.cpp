#include <stdio.h>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "GP.h"
#include "weights.h"

int main(int argc, char* argv[]){


	cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    std::vector<cv::Mat> bgr_in;
    std::vector<cv::Mat> bgr_out;
	int chan = img.channels()==3? CV_32FC3 : CV_32F;
	img.convertTo(img, chan);
	img /= 255;
    cv::split(img, bgr_in); 
	const float del[2] = {1.f/float(img.cols), 1.f/float(img.rows)}; 
	const int ratio[2] = {4, 4}; 
	weights wgts(ratio, del);
	const int size[3] = {img.cols, img.rows, img.channels()};
	GP interp(wgts, size);
    for(int i = 0; i < size[2]; i++){ 
        cv::Mat flat = bgr_in[i].reshape(1, bgr_in[i].total());
        std::vector<float> imgin = bgr_in[i].isContinuous()? flat : flat.clone();
        std::vector<float> imgout(size[0]*ratio[0]*size[1]*ratio[1], 0);
        auto start = std::chrono::high_resolution_clock::now(); 
        interp.single_channel_interp(imgin, imgout, ratio[0], ratio[1]);
        auto stop = std::chrono::high_resolution_clock::now(); 
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
        std::cout<< " Interp profile = " << duration.count() << " ms" << std::endl;
	    bgr_out.push_back(cv::Mat(imgout).reshape(1, size[1]*ratio[1]));
        bgr_out[i]*=255;
        bgr_out[i].convertTo(bgr_out[i], CV_8U); 
    }
    cv::Mat img2;
    std::cout<<"Merge"<<std::endl; 
    cv::merge(bgr_out, img2); 
    std::cout<<"Write"<<std::endl; 
	cv::imwrite("test2.png", img2);  
}

