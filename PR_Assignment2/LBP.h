#ifndef __FUNC_H__
#define __FUNC_H__
#include <opencv2\opencv.hpp>

using namespace cv;

extern int neighbors[8][2];
extern uchar pow2[8];
extern uchar cmp36[36];
extern uchar cmp256[256];

///@summary			Compute feature vector using LBP algorithm
///@param:input		2dim image
///@param:cmp36		a array to enable rotation invariant
///@param:cmp256	a array to enable rotation invariant
///@return:			a feature vector
Mat LBP(Mat input, uchar cmp36 [], uchar cmp256 []);

#endif