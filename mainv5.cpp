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

int TrainData = 900;
int TestData = 320;
int inputDepth = 1;
int inputWidth = 256; 
int inputHeight = 256;
// number of choice each task
int outputWidth = 10;
// A epoch means running through whole training set
int maxEpoch = 1;
// Because of multiple thread, please go to DeepLearning.h
// to change batchSize
// int batchSize = 60;
// To decide how many iter between show changes
// must be no bigger than maxIter
int show = 1;
int type = SR;

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

    ConvLayer conv(inputDepth, inputHeight, inputWidth, inputDepth, 9, 9, 64, 1, 4);
    ThirdReLULayer thiRel(64, inputHeight, inputWidth, 0);

    ConvLayer conv1(64, inputHeight, inputWidth, 64, 1, 1, 32, 1, 0);
    ThirdReLULayer thiRel1(32, inputHeight, inputWidth, 0);

    ConvLayer conv2(32, inputHeight, inputWidth, 32, 5, 5, inputDepth, 1, 2);
    ThirdReLULayer thiRel2(inputDepth, inputHeight, inputWidth, 1);

    OutputPictureLayer out(fpTrainLabel,fpTestLabel,TrainData,TestData,inputDepth,inputHeight,inputWidth);

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
        double oldTotalRMSE = 0;
        for(int i = 0;i < maxIter;i++){
            if((!debugMode) && (!quiet) && (i % show == 0) && (i != 0)){
                printf("%s",strShow);
                fflush(stdout);
            }

            tar = i * batchSize;
            inp.forward(tar);
            conv.forward(inp.output);
            thiRel.forward(conv.output);
            conv1.forward(thiRel.output);
            thiRel1.forward(conv1.output);
            conv2.forward(thiRel1.output);
            thiRel2.forward(conv2.output);
            out.forward(tar,thiRel2.output);

            out.backward(tar);
            thiRel2.backward(out.error);
            conv2.backward(thiRel2.error);
            thiRel1.backward(conv2.error);
            conv1.backward(thiRel1.error);
            thiRel.backward(conv1.error);
            conv.backward(thiRel.error);

            if(debugMode && (i % show == (show - 1))){
                double showLoss = (out.totalTrainLoss - oldTotalLoss) / show;
                double showRMSE = (out.totalTrainRMSE - oldTotalRMSE) / show;
                clock_t iter = clock();
                double newIter = (double)(iter - begin) / CLOCKS_PER_SEC;
                double usedTime = newIter - iterTime;
                printf("in iteration %d: ",i);
                printf("Training Loss = %llf\t",showLoss);
                printf("Training RMSE = %llf\t",showRMSE);
                printf("Used Time = %llf\n", usedTime);
                fprintf(fpResult,"in iteration %d: ",i);
                fprintf(fpResult,"Training Loss = %llf\t",showLoss);
                fprintf(fpResult,"Training RMSE = %llf\t",showRMSE);
                fprintf(fpResult,"Used Time = %llf\n", usedTime);
                fprintf(fpDebug,"%llf, %llf, %llf; ", showLoss, showRMSE, usedTime);
                fflush(fpDebug);
                fflush(fpResult);
                oldTotalLoss = out.totalTrainLoss;
                oldTotalRMSE = out.totalTrainRMSE;
                iterTime = newIter;
            }
        }

        // Testing
        int testIter = TestData / batchSize;
        if(debugMode){
            printf("Testing : ");
            fflush(stdout);
        }

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
            conv1.forward(thiRel.output);
            thiRel1.forward(conv1.output);
            conv2.forward(thiRel1.output);
            thiRel2.forward(conv2.output);
            out.TestForward(tar,thiRel2.output);
        }  

        out.ComputeTestLoss();

        if(!quiet){
            clock_t epoch = clock();
            double newEpoch = (double)(epoch - begin) / CLOCKS_PER_SEC;
            double usedTime = newEpoch - epochTime;
            printf("\nTraining Loss : %llf\n",out.totalTrainLoss / maxIter);
            fprintf(fpResult, "Training Loss : %llf\n",out.totalTrainLoss / maxIter);
            printf("Test Loss : %llf\n",out.totalTestLoss / testIter);
            fprintf(fpResult, "Test Loss : %llf\n",out.totalTestLoss / testIter);
            printf("Training RMSE : %llf\n",out.totalTrainRMSE / maxIter);
            fprintf(fpResult, "Training RMSE : %llf\n",out.totalTrainRMSE / maxIter);
            printf("Test RMSE : %llf\n",out.totalTestRMSE / testIter);
            fprintf(fpResult, "Test RMSE : %llf\n",out.totalTestRMSE / testIter);
            printf("Used Time = %llf\n", usedTime);
            fprintf(fpResult, "Used Time = %llf\n", usedTime);
            fflush(fpResult);
            epochTime = newEpoch;
        } 
        iterTime = epochTime;
        out.totalTrainLoss = 0;     
        out.totalTrainRMSE = 0; 
    }

    // Output valid data
    FILE* valid = NULL;
    fopen_s(&valid, "./sr/validResult/validResult.txt","w");
    if(valid == NULL){
        printf("Error: Unable to open valid file\n");
        fflush(stdout);
        exit(0);
    }
    out.testPredict[0].PrintMatlabArray(valid, string("boat"));
    out.testPredict[1].PrintMatlabArray(valid, string("cameraman"));
    out.testPredict[2].PrintMatlabArray(valid, string("fruits"));
    out.testPredict[3].PrintMatlabArray(valid, string("testSet"));

    return 0;
}