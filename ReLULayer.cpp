#include "DeepLearning.h"

ReLULayer::ReLULayer(int h, int w){
    height = h;
    width = w;
    for(int i = 0;i < batchSize;i++){
        Array tmp(height,width);
        output.push_back(tmp);
        error.push_back(tmp);
    }
}

double ReLULayer::ReLU(double x){
    return (x<0 ? leaky*x : x);
}

double ReLULayer::ReLUDer(double x){
    return (x<0 ? leaky : 1);
}

void ReLULayer::forward(const vector<Array>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute ReLU Forward\n");
            exit(0);
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output[n].arr[i][j] = ReLU(inp[n].arr[i][j]);
            }
        }
        if(!output[n].CheckFinite()){
            fprintf(fpDebug, "In ReLuLayer: Array called output\n");
            fflush(fpDebug);
            exit(0);
        }
    }
    return;
}

void ReLULayer::backward(const vector<Array>& err){
    for(int n = 0;n < batchSize;n++){
        if(err[n].width != width || err[n].height != height){
            printf("Error : Unable to compute ReLU backward\n");
            exit(0);
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                error[n].arr[i][j] = ReLUDer(output[n].arr[i][j]);
            }
        }
        if(!error[n].CheckFinite()){
            fprintf(fpDebug, "In ReLuLayer: Array called error\n");
            fflush(fpDebug);
            exit(0);
        }
        error[n] = DotProduct(error[n],err[n]);
        error[n].Bound(minStep,maxStep);
    }
    return;
}