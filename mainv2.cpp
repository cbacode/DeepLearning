#include "DeepLearning.h"

// Setting Epochs, LearningRate and so on
bool quiet = false;
bool debugMode = true;
double LearningRate = 0.001;
double LearningRatio = 0.95;
int LearningChange = 1;

// we use adam to fasten training
// will use corrected error
bool adam = true;
double momCoff = 0.9;
double rmsCoff = 0.99;

double minStep = eps;
double maxStep = 1e5;
double regular = 1e-4;

int TrainData = 60000;
int TestData = 10000;
int inputWidth = 28; 
int inputHeight = 28;
// number of choice each task
int outputWidth = 10;
// A epoch means running through whole training set
int maxEpoch = 4;
// Because of multiple thread, please go to DeepLearning.h
// to change batchSize
// int batchSize = 60;
// To decide how many iter between show changes
// must be no bigger than maxIter
int show = 20;
int type = MNIST;

double pic[maxWidth][maxHeight];
int label[maxNumLabel];
FILE* fpDebug = NULL;
FILE* fpResult = NULL;
ifstream* fpTrain = NULL;
ifstream* fpTest = NULL;
ifstream* fpTrainLabel = NULL;
ifstream* fpTestLabel = NULL;

vector<int> ran;
thread thr[batchSize];
const char strShow[2] = "+";

int main(){   
    readData(type);
    InputLayer inp(fpTrain,fpTest,TrainData,TestData,1,inputHeight,inputWidth);
    ConvLayer conv(1, inputHeight, inputWidth, 1, 3, 3, 4, 1, 1);
    ThirdReLULayer thiRel(4, inputHeight, inputWidth, 0);
    MaxPoolingLayer pool(4, inputHeight, inputWidth, 4, 2, 2, 2, 2);
    ThirdSpanLayer span(4, inputHeight / 2, inputWidth / 2);
    LinearLayer line(inputHeight * inputWidth, 128);
    ReLULayer rel(128, 1);
    LinearLayer line2(128, outputWidth);
    SoftMaxLayer soft(outputWidth, 1);
    OutputLayer out(fpTrainLabel,fpTestLabel,TrainData,TestData,outputWidth);

    if(!quiet){
        printf("Training...\n");
        fprintf(fpResult, "Training...\n");
        fflush(fpResult);
    }

    int tar = 0;
    clock_t begin = clock();
    double iterTime = 0;
    double epochTime = 0;
    for(int k = 0;k < maxEpoch;k++){
        // Training
        if(!quiet){
            if(!quiet) printf("Epochs %d : ",k);
            if(debugMode) printf("\n");
            if(!quiet) fprintf(fpResult,"Epochs %d : ",k);
            if(debugMode) fprintf(fpResult,"\n");
            fflush(stdout);
        }
        DecreaseLearningRate(k);
        inp.InitShuffle();

        // Number of iters in an epoch
        int maxIter = TrainData / batchSize;
        double oldTotalLoss = 0;
        for(int i = 0;i < maxIter;i++){
            if((!debugMode) && (!quiet) && (i % show == 0) && (i != 0)){
                printf("%s",strShow);
                fflush(stdout);
            }

            tar = i * batchSize;
            inp.forward(tar);
            conv.forward(inp.output);
            thiRel.forward(conv.output);
            pool.forward(thiRel.output);
            span.forward(pool.output);
            line.forward(span.output);
            rel.forward(line.output);
            line2.forward(rel.output);
            soft.forward(line2.output);
            out.forward(tar,soft.output);

            out.backward(tar);
            soft.backward(out.error);
            line2.backward(soft.error);
            rel.backward(line2.error);
            line.backward(rel.error);
            span.backward(line.error);
            pool.backward(span.error);
            thiRel.backward(pool.error);
            conv.backward(thiRel.error);

            if(debugMode && (i % show == (show - 1))){
                double showLoss = (out.totalLoss - oldTotalLoss) / show;
                double showAcc = out.accuracy / (show * batchSize);
                clock_t iter = clock();
                double newIter = (double)(iter - begin) / CLOCKS_PER_SEC;
                double usedTime = newIter - iterTime;
                printf("in iteration %d:\t",i);
                printf("Training Loss = %llf ",showLoss);
                printf("Train Accuracy = %llf ",showAcc);
                printf("Used Time = %llf\n", usedTime);
                fprintf(fpResult,"in iteration %d: ",i);
                fprintf(fpResult,"Training Loss = %llf ",showLoss);
                fprintf(fpResult,"Train Accuracy = %llf ",showAcc);
                fprintf(fpResult,"Used Time = %llf\n", usedTime);
                fprintf(fpDebug,"%llf, %llf, %llf; ", showLoss, showAcc, usedTime);
                fflush(fpDebug);
                fflush(fpResult);
                oldTotalLoss = out.totalLoss;
                iterTime = newIter;
                out.accuracy = 0;
            }
        }

        // Testing
        int testIter = TestData / batchSize;
        if(debugMode){
            printf("Testing : ");
            fflush(stdout);
        }
        out.totalLoss = 0;
        for(int i = 0;i < maxIter;i++){
            if((!quiet) && (i % show == 0) && (i != 0)){
                printf("%s",strShow);
                fflush(stdout);
            }

            tar = i * batchSize;
            inp.forward(tar);
            conv.forward(inp.output);
            thiRel.forward(conv.output);
            pool.forward(thiRel.output);
            span.forward(pool.output);
            line.forward(span.output);
            rel.forward(line.output);
            line2.forward(rel.output);
            soft.forward(line2.output);
            out.forward(tar,soft.output);
            out.backward(tar);
        }

        out.ComputeTrainAccuracy();
        double trainAcc = out.accuracy;

        for(int i = 0;i < testIter;i++){
            if(i * batchSize == TestData){
                break;
            }
            if((!quiet) && (i % show == 0) && (i != 0)){
                printf("%s",strShow);
                fflush(stdout);
            }
            
            tar = i * batchSize;
            inp.StartTest(tar);
            conv.forward(inp.output);
            thiRel.forward(conv.output);
            pool.forward(thiRel.output);
            span.forward(pool.output);
            line.forward(span.output);
            rel.forward(line.output);
            line2.forward(rel.output);
            soft.forward(line2.output);
            out.TestForward(tar,soft.output);
        }  

        out.ComputeAccuracy();

        if(!quiet){
            clock_t epoch = clock();
            double newEpoch = (double)(epoch - begin) / CLOCKS_PER_SEC;
            double usedTime = newEpoch - epochTime;
            printf("\nTraining Accuracy : %llf\n",trainAcc);
            fprintf(fpResult, "\nTraining Accuracy : %llf\n",trainAcc);
            printf("Training Loss : %llf\n",out.totalLoss / maxIter);
            fprintf(fpResult, "Training Loss : %llf\n",out.totalLoss / maxIter);
            printf("Test accuracy : %llf\n",out.accuracy);
            fprintf(fpResult, "Test accuracy : %llf\n",out.accuracy);
            printf("Used Time = %llf\n", usedTime);
            fflush(fpResult);
            epochTime = newEpoch;
        } 
        iterTime = epochTime;
        out.totalLoss = 0;     
    }
}