#include <opencv2\opencv.hpp>
#include "LBP.h"
using namespace cv;

int neighbors[8][2] = {
	-1, -1,
	-1, 0,
	-1, 1,
	0, -1,
	0, 1,
	1, -1,
	1, 0,
	1, 1 };
uchar pow2[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
uchar cmp36[36] = {
	0,
	1,
	3,
	5,
	7,
	9,
	11,
	13,
	15,
	17,
	19,
	21,
	23,
	25,
	27,
	29,
	31,
	37,
	39,
	43,
	45,
	47,
	51,
	53,
	55,
	59,
	61,
	63,
	85,
	87,
	91,
	95,
	111,
	119,
	127,
	255};
uchar cmp256[256] = {
	0, 1, 1, 3, 1, 5, 3, 7, 1, 9, 5, 11, 3, 13, 7, 15,
	1, 17, 9, 19, 5, 21, 11, 23, 3, 25, 13, 27, 7, 29, 15, 31,
	1, 9, 17, 25, 9, 37, 19, 39, 5, 37, 21, 43, 11, 45, 23, 47,
	3, 19, 25, 51, 13, 53, 27, 55, 7, 39, 29, 59, 15, 61, 31, 63,
	1, 5, 9, 13, 17, 21, 25, 29, 9, 37, 37, 45, 19, 53, 39, 61,
	5, 21, 37, 53, 21, 85, 43, 87, 11, 43, 45, 91, 23, 87, 47, 95,
	3, 11, 19, 27, 25, 43, 51, 59, 13, 45, 53, 91, 27, 91, 55, 111,
	7, 23, 39, 55, 29, 87, 59, 119, 15, 47, 61, 111, 31, 95, 63, 127,
	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
	9, 25, 37, 39, 37, 43, 45, 47, 19, 51, 53, 55, 39, 59, 61, 63,
	5, 13, 21, 29, 37, 45, 53, 61, 21, 53, 85, 87, 43, 91, 87, 95,
	11, 27, 43, 59, 45, 91, 91, 111, 23, 55, 87, 119, 47, 111, 95, 127,
	3, 7, 11, 15, 19, 23, 27, 31, 25, 39, 43, 47, 51, 55, 59, 63,
	13, 29, 45, 61, 53, 87, 91, 95, 27, 59, 91, 111, 55, 119, 111, 127,
	7, 15, 23, 31, 39, 47, 55, 63, 29, 61, 87, 95, 59, 111, 119, 127,
	15, 31, 47, 63, 61, 95, 111, 127, 31, 63, 95, 127, 63, 127, 127, 255};

uchar logic(int input)
{
	return input > 0 ? 1 : 0;
}

Mat LBP(Mat input, uchar cmp36[], uchar cmp256[])
{
	Mat intermedia = Mat::zeros(input.rows, input.cols, CV_8U);
	int row_max = input.rows - 1;
	int col_max = input.cols - 1;

	//LBP core
	uchar center, sum_;
	for (int i = 1; i < row_max; i++)
	{
		for (int j = 1; j < col_max; j++)
		{
			center = input.at<uchar>(i, j);
			sum_ = 0;
			for (int k = 7; k >= 0; k--)
				sum_ = sum_ + logic((int)input.at<uchar>(i + neighbors[7 - k][0], j + neighbors[7 - k][1]) - (int)center) * pow2[k];
			intermedia.at<uchar>(i, j) = cmp256[sum_];
		}
	}

	//histogram
	int output_[256] = { 0 };
	uchar index;
	for (int i = 1; i < row_max; i++)
	{
		for (int j = 1; j < col_max; j++)
		{
			index = intermedia.at<uchar>(i, j);
			output_[index] += 1;
		}
	}

	//compress to 36 dim
	Mat output = Mat::zeros(36, 1, CV_32F);
	int i = 0;
	for (int j = 0; j < 256; j++)
	{
		if (j == cmp36[i])
		{
			output.at<float>(i, 0) = output_[j];
			i++;
			//printf("%d:%d\n", i, output_[j]);
		}
	}

	//normalize to range [0.0, 1.0]
	Mat t;
	normalize(output, t, 1.0, 0.0, NORM_MINMAX);
	return t;
}