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

OutputPictureLayer::OutputPictureLayer(ifstream* fpTra, ifstream* fpTes, int numTrain, int numTest,int od,int oh,int ow){
    outDepth = od;
    outHeight = oh;
    outWidth = ow;
    numTrainData = numTrain;
    numTestData = numTest;

    //loss will be compute while backward propagation
    trainLoss.resize(numTrainData);
    testLoss.resize(numTestData);
    trainRMSE.resize(numTrainData);
    testRMSE.resize(numTestData);

    ThirdArray tmp(outDepth,outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        error.push_back(tmp);
    }
    for(int i = 0;i < numTrainData;i++){
        trainPredict.push_back(tmp);
    }
    for(int i = 0;i < numTestData;i++){
        testPredict.push_back(tmp);
    }
    
    for(int i = 0;i < numTrainData;i++){
        GetPicture(fpTra);
        ThirdArray dat(outDepth,outHeight,outWidth);
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    dat.thiArr[j].arr[k][l] = (double)pic[k][l] / 256.0;
                }
            }
        }
        trainTruth.push_back(dat);
    }
    if(!quiet){
        printf("Training label Successfully loaded...\n");
        fprintf(fpResult, "Training label Successfully loaded...\n");
        fflush(fpResult);
    }

    for(int i = 0;i < numTestData;i++){
        GetPicture(fpTes);
        ThirdArray dat(outDepth,outHeight,outWidth);
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    dat.thiArr[j].arr[k][l] = (double)pic[k][l] / 256.0;
                }
            }
        }
        testTruth.push_back(dat);
    }
    if(!quiet){
        printf("Testing label Successfully loaded...\n");
        fprintf(fpResult, "Testing label Successfully loaded...\n");
        fflush(fpResult);
    }
    //testTruth.PrintArray(fpDebug);
}

void OutputPictureLayer::backward(int tar){
    // Use loss function of 1/N * 1/2 * (a - y)^2 
    for(int i = 0;i < batchSize;i++){
        int iter = ran[tar + i];
        error[i] = trainPredict[iter] - trainTruth[iter];
        trainLoss[iter] = DotProduct(error[i], error[i]).addTogether();
        trainRMSE[iter] = sqrt(trainLoss[iter]);
        trainLoss[iter] /= (2.0 * batchSize);
        trainRMSE[iter] /= (1.0 * batchSize);
        totalTrainLoss += trainLoss[iter];
        totalTrainRMSE += trainRMSE[iter];
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
    for(int j = 0;j < batchSize;j++){
        if(!isfinite(trainLoss[j])){
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

void OutputPictureLayer::forward(int tar,const vector<ThirdArray>& inp){
    for(int i = 0;i < batchSize;i++){
        if(tar + i > TrainData){
            fprintf(fpDebug,"Unable to read batch label for training.\n");
            exit(0);
        }
        int iter = ran[tar + i];
        trainPredict[iter] = inp[i];
    }
    return;
}

void OutputPictureLayer::TestForward(int tar,const vector<ThirdArray>& inp){
    for(int i = 0;i < batchSize;i++){
        if(tar + i > TestData){
            fprintf(fpDebug,"Unable to read batch label for testing.\n");
            exit(0);
        }
        testPredict[tar + i] = inp[i];
    }
    return;
}

void OutputPictureLayer::ComputeTestLoss(void){
    totalTestLoss = 0;
    totalTestRMSE = 0;
    int bound = numTestData - numTestData % batchSize;
    for(int i = 0;i < bound;i++){
        ThirdArray err = testPredict[i] - testTruth[i];
        testLoss[i] = DotProduct(err, err).addTogether();
        testRMSE[i] = sqrt(testLoss[i]);
        testLoss[i] /= (2.0 * batchSize);
        testRMSE[i] /= (1.0 * batchSize);
        totalTestLoss += testLoss[i];
        totalTestRMSE += testRMSE[i];
    }
    return;
}