#ifndef __NEURALNETWORK_H__
#define __NEURALNETWORK_H__

#include <opencv2\opencv.hpp>
using namespace cv;

#define HIDDEN_NODE_NUM 12
#define OUTPUT_NODE_NUM 2

#define LEARN_STEP 1

///@summary:		Initiate weight matrix using randomized value
///@params:input	Pointer of weight matrix
///@params:rows		Number of rows of weight matrix
///@params:cols		Number of columns of weight matrix
///@return:			void
void weightInit(float **input, int rows, int cols);

///@summary:			Compute a neural network output
///@params:featVector	A feature vector of a image
///@params:W1			Weight matrix between input nodes and hidden nodes
///@params:W2			Weight matrix between hidden nodes and output nodes
///@params:hiddenNode	A array to hold result of hidden nodes
///@params:outputNode	A array to hold result of output nodes
///@return:				void
void computeNeuralNetworkOutput(Mat &featVector, float W1[36][HIDDEN_NODE_NUM], float W2[HIDDEN_NODE_NUM][OUTPUT_NODE_NUM], float hiddenNode[], float outputNode[]);
#endif