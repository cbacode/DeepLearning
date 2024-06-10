#include "DeepLearning.h"

//    int choice;
//    int inpWidth;
//    int inpHeight;
//    int batch;
//    int numTestData;
//    double totalLoss;
//    Array trainPredict;
//    Array trainTruth;
//    Array testPredict;
//    Array testTruth;
//    Array loss;
//    Array error;

OutputLayer::OutputLayer(ifstream* fpTra, ifstream* fpTes, int numTrain, int numTest, int outWid){
    outWidth = outWid;
    numTrainData = numTrain;
    numTestData = numTest;
    //final result will be compiled for save
    trainResult.ChangeSize(numTrain,outWidth);
    trainPredict.resize(numTrain);
    trainTruth.resize(numTrain);
    testResult.ChangeSize(numTest,outWidth);
    testPredict.resize(numTest);
    testTruth.resize(numTest);

    //loss will be compute while backward propagation
    loss.resize(outWidth);

    for(int i = 0;i < batchSize;i++){
        Array tmp(outWidth,1);
        error.push_back(tmp);
    }
    
    GetLabel(fpTra,numTrain);
    for(int i = 0;i < numTrainData;i++){
        trainTruth[i] = label[i];
    }
    if(!quiet){
        printf("Training label Successfully loaded...\n");
        fprintf(fpResult, "Training label Successfully loaded...\n");
        fflush(fpResult);
    }

    GetLabel(fpTes,numTest);
    for(int i = 0;i < numTestData;i++){
        testTruth[i] = label[i];
    }
    if(!quiet){
        printf("Testing label Successfully loaded...\n");
        fprintf(fpResult, "Testing label Successfully loaded...\n");
        fflush(fpResult);
    }
    //testTruth.PrintArray(fpDebug);
}

void OutputLayer::backward(int tar){
    // Use loss function of - 1/N * y * log(a)
    // Sparse Cross-Entropy Loss Function
    for(int j = 0;j < outWidth;j++){
        loss[j] = 0;
    }

    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < outWidth;j++){
            int iter = ran[tar + i];
            double num = trainResult.arr[iter][j];
            if(abs(num) < eps){
                int sign = (num >= 0) ? 1 : -1;
                num = eps * sign;
            }
            if(abs(1 - num) < eps){
                int sign = (num >= 0) ? 1 : -1;
                num = (1 - eps) * sign;
            }
            //fprintf(fpDebug, "trainTruth[iter] = %d \t iter = %d \n",trainTruth[iter],iter);
            int tmp = (trainTruth[iter] == j) ? 1 : 0;
            error[i].arr[j][0] = (num - tmp);
            loss[j] += (-tmp) * log(num) / batchSize;
        }
        
        //error[i].PrintArray(fpDebug);
    }

    for(int j = 0;j < outWidth;j++){
        totalLoss += loss[j];
    }

    for(int i = 0;i < batchSize;i++){
        if(!error[i].CheckFinite()){
            fprintf(fpDebug, "In OutputLayer : Array called error\n");
            fflush(fpDebug);
            exit(0);
        }
        //error[i].Bound(minStep,maxStep);
    }

    bool check = true;
    for(int j = 0;j < outWidth;j++){
        if(!isfinite(loss[j])){
            check = false;
        }
    }
    if(!check){
        fprintf(fpDebug, "In OutputLayer : Array called eachLoss\n");
        fflush(fpDebug);
        exit(0);
    }
    return;
}

void OutputLayer::forward(int tar,const vector<Array>& inp){
    int max = 0;
    for(int i = 0;i < batchSize;i++){
        if(tar + i > TrainData){
            fprintf(fpDebug,"Unable to read batch label for training.\n");
            exit(0);
        }
        int iter = ran[tar + i];
        for(int j = 0;j < outWidth;j++){
            trainResult.arr[iter][j] = inp[i].arr[j][0];
            if(trainResult.arr[iter][j] > trainResult.arr[iter][max]){
                max = j;
            }
        }
        trainPredict[iter] = max;
        if(trainPredict[iter] == trainTruth[iter]){
            accuracy += 1;
        }
    }
    return;
}

void OutputLayer::TestForward(int tar,const vector<Array>& inp){
    int max = 0;
    for(int i = 0;i < batchSize;i++){
        int iter = tar + i;
        for(int j = 0;j < outWidth;j++){
            testResult.arr[iter][j] = inp[i].arr[j][0];
            if(testResult.arr[iter][j] > testResult.arr[iter][max]){
                max = j;
            }
        }
        testPredict[iter] = max;
    }
    return;
}

void OutputLayer::ComputeAccuracy(void){
    accuracy = 0;
    for(int i = 0;i < numTestData;i++){
        if(testPredict[i] == testTruth[i]){
            accuracy += 1;
        }
    }
    accuracy /= numTestData;
    return;
}

void OutputLayer::ComputeTrainAccuracy(void){
    accuracy = 0;
    for(int i = 0;i < numTrainData;i++){
        if(trainPredict[i] == trainTruth[i]){
            accuracy += 1;
        }
    }
    accuracy /= numTrainData;
    return;
}