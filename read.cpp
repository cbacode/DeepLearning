#include "DeepLearning.h"
// type == 1 MNIST
// type == 2 FashionMNIST
// type == 3 SR
void readData(int type){
    fopen_s(&fpDebug, "./Debug.txt","w");
    if(fpDebug == NULL){
        printf("Error: Unable to open debug file\n");
        exit(0);
    }
    //TestArray(fpDebug);
    //TestSign(fpDebug);
    //TestSingleConv(fpDebug);
    //TestSpread(fpDebug);
    //TestDiffConv(fpDebug);
    //TestDiffLinear(fpDebug);
    //TestDiffBatchNorm(fpDebug);
    //TestMaxPooling(fpDebug);
    //TestThiRel(fpDebug);
   
    fopen_s(&fpResult, "./Result.txt","w");
    if(fpResult == NULL){
        printf("Error: Unable to open result file\n");
        exit(0);
    }

    switch(type){
        case MNIST:
        fpTrain = new ifstream("./MNIST/train-images.idx3-ubyte",ios::binary);
        fpTest = new ifstream("./MNIST/t10k-images.idx3-ubyte",ios::binary);
        fpTrainLabel = new ifstream("./MNIST/train-labels.idx1-ubyte",ios::binary);
        fpTestLabel = new ifstream("./MNIST/t10k-labels.idx1-ubyte",ios::binary);
        break;
        case FashionMNIST:
        fpTrain = new ifstream("./FashionMNIST/train-images-idx3-ubyte",ios::binary);
        fpTest = new ifstream("./FashionMNIST/t10k-images-idx3-ubyte",ios::binary);
        fpTrainLabel = new ifstream("./FashionMNIST/train-labels-idx1-ubyte",ios::binary);
        fpTestLabel = new ifstream("./FashionMNIST/t10k-labels-idx1-ubyte",ios::binary);
        break;
        case SR:
        fpTrain = new ifstream("./sr/train-images.bin",ios::binary);
        fpTest = new ifstream("./sr/test-images.bin",ios::binary);
        fpTrainLabel= new ifstream("./sr/train-labels.bin",ios::binary);
        fpTestLabel = new ifstream("./sr/test-labels.bin",ios::binary);
        break;
        default:
        printf("Error input type.\n");
        exit(0);
        break;
    }
    if(!fpTrain->is_open() || fpTrain == NULL){
        printf("Error: Unable to open training file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTrainingPictureFile(fpTrain);

    if(!fpTest->is_open() || fpTest == NULL){
        printf("Error: Unable to open testing file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTestingPictureFile(fpTest);

    // Check function GetData in basic.cpp to ensure
    // one gray picture or other material is loaded 
    // each time the function is called.

    if(!fpTrainLabel->is_open() || fpTrainLabel == NULL){
        printf("Error: Unable to open training label file\n");
        exit(0);
    }
    // Defining checking function of the file
    switch(type){
        case MNIST:
        case FashionMNIST:
        CheckTrainingLabelFile(fpTrainLabel);
        break;
        case SR:
        CheckTrainingPictureFile(fpTrainLabel);
        break;
    }
    
    if(!fpTestLabel->is_open() || fpTestLabel == NULL){
        printf("Error: Unable to open testing label file\n");
        exit(0);
    }
    // Defining checking function of the file
    switch(type){
        case MNIST:
        case FashionMNIST:
        CheckTestingLabelFile(fpTestLabel);
        break;
        case SR:
        CheckTestingPictureFile(fpTestLabel);
        break;
    }
}

void CheckTrainingPictureFile(ifstream* fp){
    int magic = GetNumber(fp);
    if(magic != 2051){
        printf("Warning : Loading Training picture \n");
        printf("Warning : Error magic number \n");
        printf("Warning : The number read is %d. \n", magic);
    }
    else if(!quiet){
        fprintf(fpResult, "Training magic number Loaded : The number read is %d. \n", magic);
    }
    int numImages = GetNumber(fp);
    if(numImages < TrainData){
        printf("Error : Unable to load enough training picture.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    else if(!quiet){
        fprintf(fpResult, "Training image number Loaded : The number read is %d. \n", numImages);
    }
    // The relation can be wrong, but now row = col 
    // So this is not important
    int numRows = GetNumber(fp);
    if(numRows != inputHeight){
        printf("Error : Different training picture height size.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    int numCols = GetNumber(fp);
    if(numCols != inputWidth){
        printf("Error : Different training picture width size.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    return;
}

void CheckTestingPictureFile(ifstream* fp){
    int magic = GetNumber(fp);
    if(magic != 2051){
        printf("Warning : Loading Testing picture \n");
        printf("Warning : Error magic number \n");
        printf("Warning : The number read is %d. \n", magic);
    }
    else if(!quiet){
        fprintf(fpResult, "Testing magic number Loaded : The number read is %d. \n", magic);
    }
    int numImages = GetNumber(fp);
    if(numImages < TestData){
        printf("Error : Unable to load enough testing picture.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    else if(!quiet){
        fprintf(fpResult, "Testing image number Loaded : The number read is %d. \n", numImages);
    }
    // The relation can be wrong, but now row = col 
    // So this is not important
    int numRows = GetNumber(fp);
    if(numRows != inputHeight){
        printf("Error : Different testing picture height size.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    int numCols = GetNumber(fp);
    if(numCols != inputWidth){
        printf("Error : Different testing picture width size.\n");
        printf("Warning : The number read is %d. \n", numImages);
        exit(0);
    }
    return;
}

void CheckTrainingLabelFile(ifstream* fp){
    int magic = GetNumber(fp);
    if(magic != 2049){
        printf("Warning : Loading Training label \n");
        printf("Warning : Error magic number \n");
        printf("Warning : The number read is %d. \n", magic);
    }
    else if(!quiet){
        fprintf(fpResult, "Training label magic number Loaded : The number read is %d.\n", magic);
    }
    int numLabels = GetNumber(fp);
    if(numLabels < TrainData){
        printf("Error : Unable to load enough training label.\n");
        printf("Warning : The number read is %d. \n", numLabels);
        exit(0);
    }
    else if(!quiet){
        fprintf(fpResult, "Training label number Loaded : The number read is %d. \n", numLabels);
    }
    return;
}

void CheckTestingLabelFile(ifstream* fp){
    int magic = GetNumber(fp);
    if(magic != 2049){
        printf("Warning : Loading Testing label \n");
        printf("Warning : Error magic number \n");
        printf("Warning : The number read is %d. \n", magic);
    }
    else if(!quiet){
        fprintf(fpResult, "Testing magic number Loaded : The number read is %d. \n", magic);
    }
    int numLabels = GetNumber(fp);
    if(numLabels < TestData){
        printf("Error : Unable to load enough testing label.\n");
        printf("Warning : The number read is %d. \n", numLabels);
        exit(0);
    }
    else if(!quiet){
        fprintf(fpResult, "Testing label number Loaded : The number read is %d. \n", numLabels);
    }
    return;
}