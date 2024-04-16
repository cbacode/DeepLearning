#pragma once
#include<iostream>
#include<stdio.h>
#include<vector>
#include<time.h>
#include<random>
#include<cmath>
#include<float.h>
#include<fstream>
using namespace std;

extern bool quiet;
extern double LearningRate;
extern double LearningRatio;
extern int LearningChange;
extern int TrainData;
extern int TestData;
extern int inputWidth; 
extern int inputHeight;
extern int outputWidth;
extern double minStep;
extern double maxStep;
extern double maxInput;

extern FILE* fpDebug;
extern FILE* fpResult;
extern ifstream* fpTrain;
extern ifstream* fpTest;
extern ifstream* fpTrainLabel;
extern ifstream* fpTestLabel;

const int maxWidth = 30;
const int maxHeight = 30;
const int maxNumLabel = 70000;
const double eps = 1e-7;
extern double pic[maxWidth][maxHeight];
extern int label[maxNumLabel];
extern vector<int> ran;

class Array{
    public:
    vector<vector<double> > arr;
    int width;
    int height;
    Array Transfer();
    void LineProduct(int h,Array* from,Array* tar);
    void ChangeSize(int h,int w);
    Array(int wid,int hei);
    Array(void); 
    void Bound(double mini,double maxi);
    bool CheckFinite(void);
    void operator =(Array right);
    void PrintArray(FILE* fpDebug);  
};

class LinearLayer{
    public:
    int inpSize;
    int outSize;
    int width;
    Array inp;
    Array coff;
    Array bias;
    Array output;
    Array error;
    LinearLayer(int inp, int out, int w);
    void InitCoff(void);
    void InitBias(void);
    // Use Normalization function
    void forward(Array* input);
    void backward(Array* err);
};

class SigmoidLayer{
    public:
    int height;
    int width;
    Array output;
    Array error;
    SigmoidLayer(int h,int w);
    double Sigmoid(double x);
    void forward(Array* input);
    void backward(Array* err);
};

class ReLULayer{
    public:
    int height;
    int width;
    Array output;
    Array error;
    double leaky = 0.01;
    ReLULayer(int h, int w);
    double ReLU(double x);
    double ReLUDer(double x);
    void forward(Array* input);
    void backward(Array* err);
};

class ConvLayer{
    public:
    int outHeight;
    int outWidth;
    int height;
    int width;
    Array kernal;
    Array output;
    Array error;
    ConvLayer(int oh, int ow, int h, int w);
    void InitKernel(void);
    // Use Normalization function
    void forward(Array* input);
    void backward(Array* err);
};

class SpanLayer{
    public:
    Array output;
    Array error;
    int height;
    int width;
    SpanLayer(int height,int width);
    void forward(Array* input);
    void backward(Array* err);
};

class SoftMaxLayer{
    public:
    Array input;
    Array output;
    Array error;
    int height;
    int width;
    vector<double> sum;
    SoftMaxLayer(int height,int width);
    void SoftMaxDer(int h,int w,Array *err);
    void AddTogether(void);
    void forward(Array* input);
    void backward(Array* err);
};

class InputLayer{
    public:
    vector<Array> trainData;
    vector<Array> testData;
    int numBatch;
    int numTrainData;
    int numTestData;
    int inpWidth;
    int inpHeight;
    Array output;
    InputLayer(ifstream* fpTrain, ifstream* fpTest, int batch, int train, int test, int height, int width);
    void InitShuffle(void);
    void StartTest(int epoch);
    // For testing
    // void Generalize(Array* inp);
    // restrict maximum number to no more than one
    void forward(int epoch);
    // For training
};

class OutputLayer{
    public:
    int outWidth;
    int batchSize;
    int numTrainData;
    int numTestData;
    vector<int> trainPredict;
    vector<int> trainTruth;
    vector<int> testPredict;
    vector<int> testTruth;
    Array trainResult;
    Array testResult;
    Array error;
    vector<double> loss;
    double totalLoss;
    double accuracy;
    OutputLayer(ifstream* fpTrain, ifstream* fpTest, int batch, int numTrain, int numTest, int outWidth);
    void backward(int iter);
    void forward(int iter,Array* res);
    void TestClassify(int iter,Array* res);
    void ComputeAccuracy(void);
};

// basic.cpp
int GetNumber(ifstream* fp);
void GetPicture(ifstream* fp);
void GetLabel(ifstream* fp, int len);
void DecreaseLearningRate(int epoch);

void TestArray(FILE* fpDebug);
void TestSign(FILE* fpDebug);

// Array.cpp
Array DotProduct(Array left,Array right);
Array operator +(Array left,Array right);
Array operator *(Array left,Array right);
Array operator *(double left, Array right);

// read.cpp
void readData(void);
void CheckTrainingPictureFile(ifstream* fp);
void CheckTestingPictureFile(ifstream* fp);
void CheckTrainingLabelFile(ifstream* fp);
void CheckTestingLabelFile(ifstream* fp);