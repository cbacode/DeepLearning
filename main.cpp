#include "DeepLearning.h"

bool quiet = false;
double LearningRate = 0.001;
double LearningRatio = 0.9;
int LearningChange = 1000;

double minStep = eps;
double maxStep = 1e7;
double maxInput = 5e2;

int TrainData = 60000;
int TestData = 10000;
int inputWidth = 28; 
int inputHeight = 28;
// number of choice each task
int outputWidth = 10;

double pic[maxWidth][maxHeight];
int label[maxNumLabel];
FILE* fpDebug = NULL;
FILE* fpResult = NULL;
ifstream* fpTrain = NULL;
ifstream* fpTest = NULL;
ifstream* fpTrainLabel = NULL;
ifstream* fpTestLabel = NULL;

vector<int> ran;

int main(){   
    // It is better to have numTrainData % batchSize == 0
    // And (maxEpoch * batchSize) / numTrainData is a integer not too big
    int maxEpoch = 40000;
    int batchSize = 60;
    // To decide how many epoch between show changes
    // must be no bigger than maxEpoch
    int divide = 100;
    // To decide how many epoch between show changes
    // must be no bigger than divide
    int show = 10;
    const char strShow[2] = "+";
    // Setting Epochs, LearningRate and so on

    readData();
    InputLayer inp(fpTrain,fpTest,batchSize,TrainData,TestData,inputHeight,inputWidth);
    //fprintf(fpDebug,"In main\n");
    //inp.trainData[40].PrintArray(fpDebug);
    SpanLayer span(inputHeight, inputWidth);
    LinearLayer line(inputHeight * inputWidth, 128, 1);
    ReLULayer rel(128, 1);
    LinearLayer line2(128, outputWidth, 1);
    SoftMaxLayer soft(outputWidth, 1);
    OutputLayer out(fpTrainLabel,fpTestLabel,batchSize,TrainData,TestData,outputWidth);

    //inp.Generalize(&train); 
    //exit(0);
    if(!quiet){
        printf("Training...\n");
        fprintf(fpResult, "Training...\n");
        fflush(fpResult);
    }

    if(maxEpoch * batchSize < TrainData){
        printf("Error : Bad maxEpoch chosen.\n");     
        exit(0);  
    }
    if(batchSize > TrainData){
        printf("Error : Bad batchSize chosen.\n");     
        exit(0);  
    }
    for(int i = 0;i < maxEpoch;i++){
        DecreaseLearningRate(i);
        //inp.InitShuffle();

        if(!quiet && i % divide == 0){
            if(!quiet) printf("Epochs %d : ",i);
            if(!quiet) fprintf(fpResult,"Epochs %d : ",i);
            //printf("Computing forward propagation : ");
            fflush(stdout);
        }

        if((!quiet) && i % show == 0){
            printf("%s",strShow);
            fflush(stdout);
        }
        for(int j = 0;j < batchSize;j++){
            //int tar = (i * batchSize + j) % TrainData;
            //fprintf(fpDebug, "Before InputLayer\n");
            //fflush(fpDebug);
            int tar = (i * batchSize + j) % TrainData;
            if(tar == 0 && j != 0){
                printf("Error : Unexpected time to shuffle.\n");
                exit(0);
            }
            inp.forward(tar);
            //fprintf(fpDebug, "tar = %d\n",tar);
            //fprintf(fpDebug, "Before SpanLayer\n");
            //fflush(fpDebug);
            span.forward(&inp.output);
            //fprintf(fpDebug, "Before LineLayer\n");
            //fflush(fpDebug);
            line.forward(&span.output);
            //fprintf(fpDebug, "Before SigmoidLayer\n");
            //fflush(fpDebug);
            rel.forward(&line.output);
            //fprintf(fpDebug, "Before SoftMaxLayer\n");
            //fflush(fpDebug);
            line2.forward(&rel.output);
            soft.forward(&line2.output);
            //fprintf(fpDebug, "Before OutputLayer\n");
            //fflush(fpDebug);
            out.forward(tar,&soft.output);
            //print train accuracy
        }
        //out.trainResult.PrintArray(fpDebug);

        if(!quiet && i % divide == divide - 1) printf("\n");
        // if(!quiet){
        //     printf("Computing backward propagation : ");
        //     fflush(stdout);
        // }
        
        int tar = (i * batchSize) % TrainData;
        //fprintf(fpDebug, "Before OutputLayer\n");
        //fflush(fpDebug);
        out.backward(tar);
        //fprintf(fpDebug, "out.tar = %d \n",inp.shuffle[tar]);
        //fprintf(fpDebug, "Before SoftMaxLayer\n");
        //fflush(fpDebug);
        soft.backward(&out.error);
        line2.backward(&soft.error);
        //fprintf(fpDebug, "Before SigmoidLayer\n");
        //fflush(fpDebug);
        rel.backward(&line2.error);
        //fprintf(fpDebug, "Before LinearLayer\n");
        //fflush(fpDebug);
        line.backward(&line2.error);
        //line.backward(&out.error);
        //fprintf(fpDebug, "Before SpanLayer\n");
        //fflush(fpDebug);
        //span.backward(&line.error);
        
        //if(!quiet) printf("Done. \n");

        //out.ComputeLoss();
        if(!quiet && i % divide == divide - 1){
            printf("Training Loss : %llf\n",out.totalLoss);
            fprintf(fpResult, "Training Loss : %llf\n",out.totalLoss);

            printf("Accuracy : %llf\n",out.accuracy);
            fprintf(fpResult, "Accuracy : %llf\n",out.accuracy);
            fflush(fpResult);
        }
    }
    
    if(!quiet) printf("\n");

    if(!quiet){
        printf("Computing test results : \n");
        fflush(stdout);
    }
    int testEpoch = TestData / batchSize + 1;
    for(int i = 0;i < testEpoch;i++){
        if(i * batchSize == TestData){
            break;
        }
        if(!quiet && i % divide == 0) printf("Epochs %d : ",i);
        if((!quiet) && i % show == 0){
            printf("%s",strShow);
            fflush(stdout);
        }
        for(int j = 0;j < batchSize;j++){
            if(i * batchSize + j == TestData){
                printf("\n");
                break;
            }
            
            //fprintf(fpDebug, "Before InputLayer\n");
            inp.StartTest(i * batchSize + j);
            span.forward(&inp.output);
            //fprintf(fpDebug, "Before LinearLayer\n");
            line.forward(&span.output);
            //fprintf(fpDebug, "Before SigmoidLayer\n");
            rel.forward(&line.output);
            line2.forward(&rel.output);
            soft.forward(&line2.output);
            //fprintf(fpDebug, "Before OutputLayer\n");
            out.TestClassify(i * batchSize + j ,&soft.output);
        }
        if(!quiet && i % divide == divide - 1) printf("\n");
    }   
    out.ComputeAccuracy();

    // fprintf(fpDebug,"Train Data\n");
    // for(int i = 0;i < TrainData;i++){
    //     fprintf(fpDebug,"%d %d\n",out.trainPredict[i],out.trainTruth[i]);
    // }
    // fprintf(fpDebug,"Test Data\n");
    // for(int i = 0;i < TestData;i++){
    //     fprintf(fpDebug,"%d %d\n",out.testPredict[i],out.testTruth[i]);
    // }
    if(!quiet){
        printf("Test accuracy : %llf\n",out.accuracy);
        fprintf(fpResult, "Test accuracy : %llf\n",out.accuracy);
        fflush(fpResult);
    }
}