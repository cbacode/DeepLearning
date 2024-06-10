#pragma once
#include<iostream>
#include<stdlib.h>
#include<stdio.h>
#include<vector>
#include<time.h>
#include<chrono>
#include<random>
#include<cmath>
#include<float.h>
#include<fstream>
#include<thread>
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
extern int maxEpoch;
// extern int batchSize;
extern int show;
extern double minStep;
extern double maxStep;
extern double momCoff;
extern double rmsCoff;
extern bool adam;
extern double regular;

extern FILE* fpDebug;
extern FILE* fpResult;
extern ifstream* fpTrain;
extern ifstream* fpTest;
extern ifstream* fpTrainLabel;
extern ifstream* fpTestLabel;

const int maxWidth = 300;
const int maxHeight = 300;
const int maxNumLabel = 70000;
// change batchSize here
const int batchSize = 16;
const double eps = 1e-7;
extern double pic[maxWidth][maxHeight];
extern int label[maxNumLabel];
extern vector<int> ran;

extern thread thr[batchSize];

enum TYPE{
    MNIST,
    FashionMNIST,
    SR
};

class Array{
    public:
    vector<vector<double> > arr;
    int width;
    int height;
    Array Transfer(void);
    double addTogether(void);
    Array Rotate(void);
    Array Spread(int stride,int outHeight,int outWidth);
    void ChangeSize(int h,int w);
    Array(int wid,int hei);
    Array(void); 
    void Bound(double mini,double maxi);
    bool CheckFinite(void);
    void operator =(Array right);
    void operator =(double right);
    void PrintArray(FILE* fpDebug);  
    void PrintMatlabArray(FILE* fpDebug);  
    void PrintArrayAvg(FILE* fpDebug);
};

class ThirdArray{
    public:
    int height;
    int width;
    int depth;
    vector<Array> thiArr;
    ThirdArray(void);
    ThirdArray(int dep,int hei,int wid);
    void PrintArray(FILE* feDebug);
    void PrintMatlabArray(FILE* fpDebug, string prefix); 
    void PrintArrayAvg(FILE* fpDebug);
    double addTogether(void);
    void ChangeSize(int d,int h,int w);
    void Bound(double mini,double maxi);
    bool CheckFinite(void);
    void operator =(ThirdArray right);
    void operator =(double right);
};

class LinearLayer{
    public:
    int inpHeight;
    int inpWidth;
    int outHeight;
    int outWidth;
    int epoch;
    vector<Array> inp;
    Array coff;
    Array bias;
    vector<Array> output;
    vector<Array> error;
    // store data for momentum drop
    Array momForCoff;
    Array momForBias;
    // store data for RMSprop
    Array rmsForCoff;
    Array rmsForBias;
    Array diffBias;
    Array diffCoff;
    LinearLayer(int inp, int out);
    void InitCoff(void);
    void InitBias(void);
    // Use Normalization function
    void forward(const vector<Array>& input);
    void threadForward(int i);
    void backward(const vector<Array>& err);
    void threadBackward(int i,Array& diffBias,Array& diffCoff,const Array& err);
    void Adam(Array &var,Array& cacheMom,Array& cacheRms,Array& diff,int epoch);
};

class SigmoidLayer{
    public:
    int height;
    int width;
    vector<Array> output;
    vector<Array> error;
    SigmoidLayer(int h,int w);
    double Sigmoid(double x);
    void forward(const vector<Array>& input);
    void backward(const vector<Array>& err);
};

class ReLULayer{
    public:
    int height;
    int width;
    vector<Array> output;
    vector<Array> error;
    double leaky = 0.01;
    ReLULayer(int h, int w);
    double ReLU(double x);
    double ReLUDer(double x);
    void forward(const vector<Array>& input);
    void backward(const vector<Array>& err);
};

class ThirdReLULayer{
    public:
    int height;
    int width;
    int depth;
    int type;
    double maxi;
    double mini;
    vector<ThirdArray> output;
    vector<ThirdArray> error;
    double leaky = 0.01;
    ThirdReLULayer(int d,int h,int w,int type);
    double ReLU(double x);
    double ReLUDer(double x);
    void forward(const vector<ThirdArray>& input);
    void backward(const vector<ThirdArray>& err);
};

class ConvLayer{
    public:
    int epoch;

    int inpDepth;
    int inpHeight;
    int inpWidth;

    int outDepth;
    int outHeight;
    int outWidth;

    int kerDepth;
    int kerHeight;
    int kerWidth;

    int numKernel;
    int stride;
    int padding;

    vector<ThirdArray> kernal;
    vector<double> bias;
    vector<ThirdArray> input;
    vector<ThirdArray> output;
    vector<ThirdArray> error;

    vector<ThirdArray> momForKernal;
    vector<ThirdArray> rmsForKernal;
    vector<double> momForBias;
    vector<double> rmsForBias;
    vector<ThirdArray> diffKer;
    vector<double> diffBias;
    ConvLayer(int id,int ih,int iw,int kd,int kh,int kw,int num,int s,int pad);
    void InitKernel(void);
    // Use Normalization function
    void forward(const vector<ThirdArray>& input);
    void threadForward(int i);
    void backward(const vector<ThirdArray>& err);
    void threadBackward(vector<double>& DiffBias,vector<ThirdArray>& DiffKer,ThirdArray& singleErr,ThirdArray err,ThirdArray inp);
    void Adam(double &var,double& cacheMom,double& cacheRms,double& diff,int epoch);
    void Adam(ThirdArray &var,ThirdArray& cacheMom,ThirdArray& cacheRms,ThirdArray& diff,int epoch);
};

class MaxPoolingLayer{
    public:
    int inpDepth;
    int inpHeight;
    int inpWidth;

    int outDepth;
    int outHeight;
    int outWidth;

    int kerDepth;
    int kerHeight;
    int kerWidth;

    int stride;
    // type = 1 means MaxPooling
    // type = 2 means AvgPooling
    int type;

    vector<ThirdArray> output;
    vector<ThirdArray> error;
    vector<ThirdArray> place;

    MaxPoolingLayer(int id,int ih,int iw,int kd,int kh,int kw,int st,int ty);
    void forward(const vector<ThirdArray>& inp);
    void MaxPoolingForward(int tar,const ThirdArray& inp,int d,int h,int w);
    void AvgPoolingForward(int tar,const ThirdArray& inp,int d,int h,int w);
    void backward(const vector<ThirdArray>& err);
    void MaxPoolingBackward(int tar,const ThirdArray& inp,int d,int h,int w);
    void AvgPoolingBackward(int tar,const ThirdArray& inp,int d,int h,int w);
};

class BatchLayer{
    public:
    int epoch;
    int inpDepth;
    int inpHeight;
    int inpWidth;
    ThirdArray mu;
    ThirdArray sigma;
    ThirdArray testMu;
    ThirdArray testSigma;
    ThirdArray gamma;
    ThirdArray beta;

    ThirdArray momBeta;
    ThirdArray momGamma;
    ThirdArray rmsBeta;
    ThirdArray rmsGamma;

    vector<ThirdArray> norm;
    vector<ThirdArray> output;
    vector<ThirdArray> error;

    vector<ThirdArray> singleDiffGamma;
    vector<ThirdArray> singleDiffSigma;
    vector<ThirdArray> singleDiffMuLeft;
    vector<ThirdArray> singleDiffMuRight;

    ThirdArray diffGamma;
    ThirdArray diffBeta;

    BatchLayer(int id,int ih,int iw);
    void InitCoff(void);
    void forward(const vector<ThirdArray>& inp);
    void testForward(const vector<ThirdArray>& inp);
    void ThreadForward(int i,ThirdArray inp);
    void ThreadTestForward(int i,ThirdArray inp);
    void backward(const vector<ThirdArray>& err);
    void ThreadBackward(int i,const ThirdArray& err);
    void Adam(ThirdArray &var,ThirdArray& cacheMom,ThirdArray& cacheRms,ThirdArray& diff,int epoch);
};

class SpanLayer{
    public:
    vector<Array> output;
    vector<Array> error;
    int height;
    int width;
    SpanLayer(int height,int width);
    void forward(const vector<Array>& input);
    void backward(const vector<Array>& err);
};

class ThirdSpanLayer{
    public:
    vector<Array> output;
    vector<ThirdArray> error;
    int depth;
    int height;
    int width;
    ThirdSpanLayer(int depth,int height,int width);
    void forward(const vector<ThirdArray>& input);
    void backward(const vector<Array>& err);
};

class SoftMaxLayer{
    public:
    vector<Array> output;
    vector<Array> error;
    int height;
    int width;
    vector<double> sum;
    SoftMaxLayer(int height,int width);
    void AddTogether(int n);
    void forward(const vector<Array>& input);
    void backward(const vector<Array>& err);
};

class InputLayer{
    public:
    vector<ThirdArray> trainData;
    vector<ThirdArray> testData;
    int numBatch;
    int numTrainData;
    int numTestData;
    int outDepth;
    int outWidth;
    int outHeight;
    vector<ThirdArray> output;
    InputLayer(ifstream* fpTrain, ifstream* fpTest, int train, int test, int depth, int height, int width);
    void InitShuffle(void);
    void StartTest(int begin);
    // For testing
    // void Generalize(Array* inp);
    // restrict maximum number to no more than one
    void forward(int begin);
    // For training
};

class OutputLayer{
    public:
    int outWidth;
    int numTrainData;
    int numTestData;
    vector<int> trainPredict;
    vector<int> trainTruth;
    vector<int> testPredict;
    vector<int> testTruth;
    Array trainResult;
    Array testResult;
    vector<Array> error;
    vector<double> loss;
    double totalLoss;
    double accuracy;
    OutputLayer(ifstream* fpTrain, ifstream* fpTest, int numTrain, int numTest, int outWidth);
    void backward(int tar);
    void forward(int iter,const vector<Array>& res);
    void TestForward(int iter,const vector<Array>& res);
    void ComputeAccuracy(void);
    void ComputeTrainAccuracy(void);
};

class OutputPictureLayer{
    public:
    int outDepth;
    int outHeight;
    int outWidth;
    int numTrainData;
    int numTestData;
    vector<ThirdArray> trainPredict;
    vector<ThirdArray> trainTruth;
    vector<ThirdArray> testPredict;
    vector<ThirdArray> testTruth;

    vector<ThirdArray> error;
    vector<double> trainLoss;
    vector<double> testLoss;
    double totalTrainLoss;
    double totalTestLoss;

    vector<double> trainRMSE;
    vector<double> testRMSE;
    double totalTrainRMSE;
    double totalTestRMSE;
    OutputPictureLayer(ifstream* fpTrain,ifstream* fpTest,int numTrain,int numTest,int outDepth,int outHeight,int outWidth);
    void backward(int tar);
    void forward(int iter,const vector<ThirdArray>& res);
    void TestForward(int iter,const vector<ThirdArray>& res);
    void ComputeTestLoss(void);
};


// basic.cpp
int GetNumber(ifstream* fp);
void GetPicture(ifstream* fp);
void GetLabel(ifstream* fp, int len);
void DecreaseLearningRate(int epoch);

void TestArray(FILE* fpDebug);
void TestSign(FILE* fpDebug);
void TestSingleConv(FILE* fpDebug);
void TestSpread(FILE* fpDebug);
void TestDiffConv(FILE* fpDebug);
void TestDiffLinear(FILE* fpDebug);
void TestDiffBatchNorm(FILE* fpDebug);
void TestMaxPooling(FILE* fpDebug);
void TestThiRel(FILE* fpDebug);

// Array.cpp
Array Conv(Array inp,Array ker,int pad,int stride);
Array DotProduct(Array left,Array right);
Array sqrt(Array inp);
Array operator +(Array left,Array right);
Array operator +(Array left, double num);
Array operator -(Array left, Array right);
Array operator *(Array left,Array right);
Array operator *(double left, Array right);
Array operator /(Array left,Array right);

// ThirdArray.cpp
Array ThirdConv(ThirdArray inp,ThirdArray ker,int pad,int stride);
ThirdArray DotProduct(ThirdArray left,ThirdArray right);
ThirdArray sqrt(ThirdArray inp);
ThirdArray operator *(double left, ThirdArray right);
ThirdArray operator +(ThirdArray left, double num);
ThirdArray operator +(ThirdArray left, ThirdArray right);
ThirdArray operator -(ThirdArray left, ThirdArray right);
ThirdArray operator /(ThirdArray left, ThirdArray right);

// read.cpp
void readData(int type);
void CheckTrainingPictureFile(ifstream* fp);
void CheckTestingPictureFile(ifstream* fp);
void CheckTrainingLabelFile(ifstream* fp);
void CheckTestingLabelFile(ifstream* fp);
