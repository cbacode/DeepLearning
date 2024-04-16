#include "DeepLearning.h"

//    int height;
//    int width;
//    Array output;
//    Array error;

SigmoidLayer::SigmoidLayer(int h, int w){
    height = h;
    width = w;
    output.ChangeSize(h,w);
    error.ChangeSize(h,w);
}

double SigmoidLayer::Sigmoid(double x){
    return 1/(1+exp(-x));
}

void SigmoidLayer::forward(Array* input){
    if(input->width != width || input->height != height){
        printf("Error : Unable to compute Sigmoid Forward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            output.arr[i][j] = Sigmoid(input->arr[i][j]);
        }
    }
    if(!output.CheckFinite()){
        fprintf(fpDebug, "In SigmoidLayer: Array called output\n");
        fflush(fpDebug);
        exit(0);
    }
    return;
}

void SigmoidLayer::backward(Array* err){
    if(err->width != width || err->height != height){
        printf("Error : Unable to compute Sigmoid backward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            //double y = Sigmoid(err->arr[i][j]);
            double y = output.arr[i][j];
            error.arr[i][j] = y*(1-y);
        }
    }
    if(!error.CheckFinite()){
        fprintf(fpDebug, "In SigmoidLayer: Array called error\n");
        fflush(fpDebug);
        exit(0);
    }
    error = DotProduct(error,(*err));
    //error = LearningRate * error;
    //error.Bound(minStep,maxStep);
    //fprintf(fpDebug,"In SigmoidLayer: \n");
    //error.PrintArray(fpDebug);
    return;
}