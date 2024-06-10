#include "DeepLearning.h"

//    vector<Array> trainData;
//    vector<Array> testData;
//    vector<int> shuffle;
//    int numTrainData;
//    int numTestData;
//    int inpWidth;
//    int inpHeight;
//    Array output;

InputLayer::InputLayer(ifstream* fpTrain, ifstream* fpTest, int train, int test, int depth, int height, int width){
    numTrainData = train;
    numTestData = test;
    
    outDepth = depth;
    outHeight = height;
    outWidth = width;

    ran.resize(numTrainData);
    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp(outDepth,outHeight,outWidth);
        output.push_back(tmp);
    }

    for(int i = 0;i < train;i++){
        GetPicture(fpTrain);
        ThirdArray dat(depth,height,width);
        for(int j = 0;j < depth;j++){
            for(int k = 0;k < height;k++){
                for(int l = 0;l < width;l++){
                    dat.thiArr[j].arr[k][l] = (double)pic[k][l] / 256.0;
                }
            }
        }
        trainData.push_back(dat);
    }

    if(!quiet){
        printf("Training Data Successfully loaded...\n");
        fprintf(fpResult, "Training Data Successfully loaded...\n");
        fflush(fpResult);
    }

    for(int i = 0;i < test;i++){
        GetPicture(fpTest);
        ThirdArray dat(depth,height,width);
        for(int j = 0;j < depth;j++){
            for(int k = 0;k < height;k++){
                for(int l = 0;l < width;l++){
                    dat.thiArr[j].arr[k][l] = (double)pic[k][l] / 256.0;
                }
            }
        }
        testData.push_back(dat);
    }

    if(trainData.size() != train){
        printf("Error number of picture loaded.\n");
        fprintf(fpResult, "Testing Data Successfully loaded...\n");
        fflush(fpResult);
    }
    if(testData.size() != test){
        printf("Error number of picture loaded.\n");
        fprintf(fpResult, "Testing Data Successfully loaded...\n");
        fflush(fpResult);
    }
    if(!quiet){
        printf("Testing Data Successfully loaded...\n");
        fprintf(fpResult, "Testing Data Successfully loaded...\n");
        fflush(fpResult);
    }
}

void InputLayer::InitShuffle(void){
    for(int i = 0;i < numTrainData;i++){
        ran[i] = i;
    }
    srand(time(NULL));
    for(int i = 0;i < numTrainData;i++){
        int pla = rand()%(numTrainData - i) + i;
        int temp = ran[pla];
        ran[pla] = ran[i];
        ran[i] = temp;
    }
}

void InputLayer::StartTest(int begin){
    for(int i = 0;i < batchSize;i++){
        if(begin + i > TestData){
            fprintf(fpDebug,"Unable to load batch data for testing.\n");
            exit(0);
        }
        output[i] = testData[begin + i];
    }
}

void InputLayer::forward(int begin){
    for(int i = 0;i < batchSize;i++){
        if(begin + i > TrainData){
            fprintf(fpDebug,"Unable to load batch data for training.\n");
            exit(0);
        }
        int tar = ran[begin + i];
        output[i] = trainData[tar];
    }    
}