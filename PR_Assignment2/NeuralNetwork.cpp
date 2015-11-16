#include <cstdlib>
#include <cmath>
#include "NeuralNetwork.h"

void weightInit(float **input, int rows, int cols)
{
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			*((float*)input + cols * i + j) = (float)rand() / RAND_MAX * 2 - 1;
	}
}

void computeNeuralNetworkOutput(Mat &featVector, float W1[36][HIDDEN_NODE_NUM], float W2[HIDDEN_NODE_NUM][OUTPUT_NODE_NUM], float hiddenNode [], float outputNode [])
{
	//compute output - hidden node
	for (int j = 0; j < HIDDEN_NODE_NUM; j++)
	{
		for (int i = 0; i < 36; i++)
			hiddenNode[j] += W1[i][j] * featVector.at<float>(i);
		hiddenNode[j] = (float) 1 / (1 + exp(-hiddenNode[j]));
	}

	//compute output - output node
	for (int k = 0; k < OUTPUT_NODE_NUM; k++)
	{
		for (int j = 0; j < HIDDEN_NODE_NUM; j++)
			outputNode[k] += W2[j][k] * hiddenNode[j];
		outputNode[k] = (float) 1 / (1 + exp(-outputNode[k]));
	}
}