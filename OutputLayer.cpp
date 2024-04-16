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

OutputLayer::OutputLayer(ifstream* fpTra, ifstream* fpTes, int batch, int numTrain, int numTest, int outWid){
    outWidth = outWid;
    batchSize = batch;
    numTrainData = numTrain;
    numTestData = numTest;
    //final result will be compiled for save
    trainResult.ChangeSize(numTrain,outWidth);
    trainPredict.resize(numTrain);
    trainTruth.resize(numTrain);
    testResult.ChangeSize(numTest,outWidth);
    testPredict.resize(numTest);
    testTruth.resize(numTest);

    //loss will be compute while front propagation
    loss.resize(outWidth);

    error.ChangeSize(outWidth,1);
    
    //fprintf(fpDebug, "Before GetLabel.\n");
    GetLabel(fpTra,numTrain);
    for(int i = 0;i < numTrainData;i++){
        trainTruth[i] = label[i];
        //trainTruth[i] = 1;
    }
    if(!quiet){
        printf("Training label Successfully loaded...\n");
        fprintf(fpResult, "Training label Successfully loaded...\n");
        fflush(fpResult);
    }
    //trainTruth.PrintArray(fpDebug);

    GetLabel(fpTes,numTest);
    for(int i = 0;i < numTestData;i++){
        testTruth[i] = label[i];
        //testTruth[i] = 1;
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
        error.arr[j][0] = 0;
        loss[j] = 0;
        for(int i = 0;i < batchSize;i++){
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
            int tmp = (trainTruth[iter] == j) ? 1 : 0;
            error.arr[j][0] += (num - tmp);
            loss[j] += (-tmp) * log(num) / batchSize;
        }
        // error.arr[j][0] /= batchSize;
        // loss[j] /= batchSize;
    }

    //fprintf(fpDebug, "In OutputLayer : before totalLoss\n");
    totalLoss = 0;
    for(int j = 0;j < outWidth;j++){
        totalLoss += loss[j];
    }
    // totalLoss /= batchSize;

    accuracy = 0;
    for(int i = 0;i < batchSize;i++){
        int iter = ran[tar + i];
        if(trainPredict[iter] == trainTruth[iter]){
            accuracy += 1;
        }
    }
    accuracy /= batchSize;

    if(!error.CheckFinite()){
        fprintf(fpDebug, "In OutputLayer : Array called error\n");
        fflush(fpDebug);
        exit(0);
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

    // error = LearningRate * error;
    // fprintf(fpDebug,"In OutputLayer.error: \n");
    // error.PrintArray(fpDebug);
    error.Bound(minStep,maxStep);
    // if(tar == batchSize){
    //     fprintf(fpDebug,"loss[batchSize] = \n");
    //     for(int j = 0;j < outWidth;j++){
    //         fprintf(fpDebug,"%llf\n",loss[j]);
    //     }
    //     fprintf(fpDebug,"\nerror[batchSize] = \n");
    //     error.PrintArray(fpDebug);
    // }
    return;
}

void OutputLayer::forward(int tar,Array* res){
    int max = 0;
    int iter = ran[tar];
    for(int i = 0;i < outWidth;i++){
        trainResult.arr[iter][i] = res->arr[i][0];
        if(trainResult.arr[iter][i] > trainResult.arr[iter][max]){
            max = i;
        }
    }
    trainPredict[iter] = max;
    if(tar == batchSize){
        fprintf(fpDebug,"trainPredict[batchSize] = %d\n",trainPredict[iter]);
        fprintf(fpDebug,"trainTruth[batchSize] = %d\n",trainTruth[iter]);
        fprintf(fpDebug,"trainResult[batchSize] = \n");
        res->PrintArray(fpDebug);
    }
    //for(int i = 0;i < outWidth;i++){
    //    fprintf(fpDebug,"%llf  ",trainResult.arr[iter][i]);
    //}
    //fprintf(fpDebug,"\n");
    
    return;
}

void OutputLayer::TestClassify(int iter,Array* res){
    int max = 0;
    for(int i = 0;i < outWidth;i++){
        testResult.arr[iter][i] = res->arr[i][0];
        if(testResult.arr[iter][i] > testResult.arr[iter][max]){
            max = i;
        }
    }
    testPredict[iter] = max;
    return;
}

void OutputLayer::ComputeAccuracy(void){
    for(int i = 0;i < numTestData;i++){
        if(testPredict[i] == testTruth[i]){
            accuracy += 1;
        }
    }
    accuracy /= numTestData;
}