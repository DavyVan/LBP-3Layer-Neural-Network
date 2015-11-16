#include <opencv2\opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <memory.h>
#include <ctime>
#include "LBP.h"
#include "NeuralNetwork.h"

//define some constant variables
#define BASE_DIR "Database_PR_02/"
#define NEG_DIR "neg/neg11"
#define POS_DIR "pos/pos05"
#define TRAIN_SET_SIZE 20		//neg OR pos, totally x2

using namespace std;
using namespace cv;

float W1[36][HIDDEN_NODE_NUM] = { 0 };
float W2[HIDDEN_NODE_NUM][OUTPUT_NODE_NUM] = { 0 };

float hiddenNode[HIDDEN_NODE_NUM] = { 0 };
float outputNode[OUTPUT_NODE_NUM] = { 0 };

float deltaHidden[HIDDEN_NODE_NUM] = { 0 };
float deltaOutput[OUTPUT_NODE_NUM] = { 0 };


int main()
{
	//timer
	double totalTime;
	clock_t start, end;
	start = clock();

	//get feature vector of negative data
	Mat featVectors = Mat::zeros(36, TRAIN_SET_SIZE * 2, CV_32F);
	char* name = new char[100];
	for (int i = 0; i < TRAIN_SET_SIZE; i++)
	{
		sprintf(name, "%s%s%03d.png", BASE_DIR, NEG_DIR, i);
		Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			return 233;
		}
		LBP(img, cmp36, cmp256).copyTo(featVectors.col(i));
		printf("negative read - %d\n", i);
	}

	//get feature vector of positive data
	//Mat posVectors = Mat::zeros(36, TRAIN_SET_SIZE, CV_32F);
	for (int i = 0; i < TRAIN_SET_SIZE; i++)
	{
		sprintf(name, "%s%s%03d.png", BASE_DIR, POS_DIR, i);
		Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			return 233;
		}
		LBP(img, cmp36, cmp256).copyTo(featVectors.col(i + TRAIN_SET_SIZE));
		printf("positive read - %d\n", i);
	}

	//neural network
	//initialize weight
	weightInit((float**) W1, 36, HIDDEN_NODE_NUM);
	weightInit((float**) W2, HIDDEN_NODE_NUM, OUTPUT_NODE_NUM);

	//train
	for (int c = 0; c < TRAIN_SET_SIZE * 2; c++)
	{
		int t = c;
		c = rand() % 40;
		//compute desired output
		computeNeuralNetworkOutput(featVectors.col(c), W1, W2, hiddenNode, outputNode);

		//compute ¦Ä for output node
		float d[2];
		if (c < 20)		//negative
		{
			d[0] = 1;
			d[1] = 0;
		}
		else			//positive
		{
			d[0] = 0;
			d[1] = 1;
		}
		for (int k = 0; k < OUTPUT_NODE_NUM; k++)
		{
			deltaOutput[k] = outputNode[k] * (1 - outputNode[k]) * (d[k] - outputNode[k]);
		}

		//compute ¦Ä for hidden node
		for (int j = 0; j < HIDDEN_NODE_NUM; j++)
		{
			float sum = 0;
			for (int k = 0; k < OUTPUT_NODE_NUM; k++)
				sum += deltaOutput[k] * W2[j][k];
			deltaHidden[j] = hiddenNode[j] * (1 - hiddenNode[j]) * sum;
		}

		//update weight - hidden layer
		for (int j = 0; j < HIDDEN_NODE_NUM; j++)
		{
			for (int i = 0; i < 36; i++)
				W1[i][j] += LEARN_STEP * deltaHidden[j] * featVectors.at<float>(i, c);
		}

		//update weight - output layer
		for (int k = 0; k < OUTPUT_NODE_NUM; k++)
		{
			for (int j = 0; j < HIDDEN_NODE_NUM; j++)
				W2[j][k] += LEARN_STEP * deltaOutput[k] * hiddenNode[j];
		}

		memset(hiddenNode, 0, HIDDEN_NODE_NUM * sizeof(float));
		memset(outputNode, 0, OUTPUT_NODE_NUM * sizeof(float));
		memset(deltaHidden, 0, HIDDEN_NODE_NUM * sizeof(float));
		memset(deltaOutput, 0, OUTPUT_NODE_NUM * sizeof(float));
		c = t;
		printf("training - %d\n", c);
	}

	//test - negative
	float negCount = 0;
	for (int i = 0; i < 1000; i++)
	{
		sprintf(name, "%s%s%03d.png", BASE_DIR, NEG_DIR, i);
		Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			return 233;
		}
		computeNeuralNetworkOutput(LBP(img, cmp36, cmp256), W1, W2, hiddenNode, outputNode);
		if ((OUTPUT_NODE_NUM == 2 && outputNode[0] > outputNode[1]) || (OUTPUT_NODE_NUM == 1 && outputNode[0] >= 0.5))
		{
			negCount++;
			printf("negative test - %d - Right\n", i);
		}
		else
			printf("negative test - %d - Wrong\n", i);
	}

	//test - positive
	float posCount = 0;
	for (int i = 0; i < 1000; i++)
	{
		sprintf(name, "%s%s%03d.png", BASE_DIR, POS_DIR, i);
		Mat img = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
		if (img.empty())
		{
			return 233;
		}
		computeNeuralNetworkOutput(LBP(img, cmp36, cmp256), W1, W2, hiddenNode, outputNode);
		if ((OUTPUT_NODE_NUM == 2 && outputNode[0] < outputNode[1]) || (OUTPUT_NODE_NUM == 1 && outputNode[0] <= 0.5))
		{
			posCount++;
			printf("positive test - %d - Right\n", i);
		}
		else
			printf("positive test - %d - Wrong\n", i);
	}

	//timer
	end = clock();
	totalTime = (double) (end - start);

	//complete
	printf("All done! Accuracy: %f%%  Total time: %f\n", (negCount + posCount) / 2000 * 100, totalTime);

	free(name);
	system("pause");
	return 0;
}