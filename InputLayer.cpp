#include "DeepLearning.h"

//    vector<Array> trainData;
//    vector<Array> testData;
//    vector<int> shuffle;
//    int numTrainData;
//    int numTestData;
//    int inpWidth;
//    int inpHeight;
//    Array output;

InputLayer::InputLayer(ifstream* fpTrain, ifstream* fpTest, int batch, int train, int test, int height, int width){
    numBatch = batch;
    numTrainData = train;
    numTestData = test;
    inpWidth = width;
    inpHeight = height;

    for(int i = 0;i < train;i++){
        GetPicture(fpTrain);
        Array dat(height,width);
        dat.height = height;
        dat.width = width;
        for(int j = 0;j < height;j++){
            for(int k = 0;k < width;k++){
                dat.arr[j][k] = (double)pic[j][k] / 256.0;
            }
        }
        trainData.push_back(dat);
    }
    //fclose(fpTrain);
    //trainData[0].PrintArray(fpDebug);

    if(!quiet){
        printf("Training Data Successfully loaded...\n");
        fprintf(fpResult, "Training Data Successfully loaded...\n");
        fflush(fpResult);
    }

    for(int i = 0;i < test;i++){
        GetPicture(fpTest);
        Array dat(height,width);
        dat.height = height;
        dat.width = width;
        for(int j = 0;j < height;j++){
            for(int k = 0;k < width;k++){
                dat.arr[j][k] = (double)pic[j][k] / 256.0;
            }
        }
        testData.push_back(dat);
    }
    //fclose(fpTest);
    //testData[0].PrintArray(fpDebug);
    //fflush(fpDebug);

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

    //InitShuffle();
}

void InputLayer::InitShuffle(void){
    //fprintf(fpDebug,"Using InitShuffle\n");
    ran.resize(numTrainData);
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
    // for(int i = 0;i < numTrainData;i++){
    //     fprintf(fpDebug,"%d \n",ran[i]); 
    // }
}

void InputLayer::StartTest(int epoch){
    output = testData[epoch];
}

void InputLayer::forward(int num){
    if(num == 0){
        InitShuffle();
    }
    int tar = ran[num];
    output = trainData[tar];
}