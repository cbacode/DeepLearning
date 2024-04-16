#include "DeepLearning.h"
void readData(void){
    fopen_s(&fpDebug, "./Debug.txt","w");
    if(fpDebug == NULL){
        printf("Error: Unable to open debug file\n");
        exit(0);
    }
    //TestArray(fpDebug);
    //TestSign(fpDebug);
   
    fopen_s(&fpResult, "./Result.txt","w");
    if(fpResult == NULL){
        printf("Error: Unable to open result file\n");
        exit(0);
    }

    // MNIST
    fpTrain = new ifstream("./train-images.idx3-ubyte",ios::binary);
    // fashion MNIST
    // fpTrain = new ifstream("./train-images-idx3-ubyte",ios::binary);
    if(!fpTrain->is_open() || fpTrain == NULL){
        printf("Error: Unable to open training file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTrainingPictureFile(fpTrain);

    // MNIST
    fpTest = new ifstream("./t10k-images.idx3-ubyte",ios::binary);
    // fashion MNIST
    // fpTest = fopen("./t10k-images-idx3-ubyte","r");
    if(!fpTest->is_open() || fpTest == NULL){
        printf("Error: Unable to open testing file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTestingPictureFile(fpTest);

    // Check function GetData in basic.cpp to ensure
    // one gray picture or other material is loaded 
    // each time the function is called.

    // MNIST
    fpTrainLabel = new ifstream("./train-labels.idx1-ubyte",ios::binary);
    // fashion MNIST
    // fpTrainLabel = new ifstream("./train-labels-idx1-ubyte",ios::binary);
    if(!fpTrainLabel->is_open() || fpTrainLabel == NULL){
        printf("Error: Unable to open training label file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTrainingLabelFile(fpTrainLabel);

    // MNIST
    fpTestLabel = new ifstream("./t10k-labels.idx1-ubyte",ios::binary);
    // fashion MNIST
    // fpTestLabel = new ifstream("./t10k-labels-idx1-ubyte",ios::binary);
    if(!fpTestLabel->is_open() || fpTestLabel == NULL){
        printf("Error: Unable to open testing label file\n");
        exit(0);
    }
    // Defining checking function of the file
    CheckTestingLabelFile(fpTestLabel);
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
    //fread(&numRows,sizeof(int),1,fp);
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