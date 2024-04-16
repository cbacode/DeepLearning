#include "DeepLearning.h"

ReLULayer::ReLULayer(int h, int w){
    height = h;
    width = w;
    output.ChangeSize(h,w);
    error.ChangeSize(h,w);
}

double ReLULayer::ReLU(double x){
    return (x<0 ? leaky*x : x);
}

double ReLULayer::ReLUDer(double x){
    return (x<0 ? leaky : 1);
}

void ReLULayer::forward(Array* input){
    if(input->width != width || input->height != height){
        printf("Error : Unable to compute ReLU Forward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            output.arr[i][j] = ReLU(input->arr[i][j]);
        }
    }
    if(!output.CheckFinite()){
        fprintf(fpDebug, "In ReLuLayer: Array called output\n");
        fflush(fpDebug);
        exit(0);
    }
    return;
}

void ReLULayer::backward(Array* err){
    if(err->width != width || err->height != height){
        printf("Error : Unable to compute ReLU backward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            error.arr[i][j] = ReLUDer(output.arr[i][j]);
        }
    }
    if(!error.CheckFinite()){
        fprintf(fpDebug, "In ReLuLayer: Array called error\n");
        fflush(fpDebug);
        exit(0);
    }
    error = DotProduct(error,(*err));
    //error = LearningRate * error;
    error.Bound(minStep,maxStep);
    return;
}