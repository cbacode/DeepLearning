#include "DeepLearning.h"

ThirdReLULayer::ThirdReLULayer(int d,int h,int w,int t){
    height = h;
    width = w;
    depth = d;
    type = t;
    maxi = 1;
    mini = 0;
    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp(depth,height,width);
        output.push_back(tmp);
        error.push_back(tmp);
    }
}

double ThirdReLULayer::ReLU(double x){
    if(type == 1 && x > maxi) return maxi;
    if(type == 1 && x < mini) return mini;
    return (x<0 ? leaky*x : x);
}

double ThirdReLULayer::ReLUDer(double x){
    if(type == 1 && x > maxi) return 0;
    if(type == 1 && x < mini) return 0;
    return (x<0 ? leaky : 1);
}

void ThirdReLULayer::forward(const vector<ThirdArray>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].depth != depth || inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute ThirdReLU Forward.\n");
            exit(0);
        }
        for(int k = 0;k < depth;k++){
            for(int i = 0;i < height;i++){
                for(int j = 0;j < width;j++){
                    output[n].thiArr[k].arr[i][j] = ReLU(inp[n].thiArr[k].arr[i][j]);
                }
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

void ThirdReLULayer::backward(const vector<ThirdArray>& err){
    for(int n = 0;n < batchSize;n++){
        if(err[n].depth != depth || err[n].width != width || err[n].height != height){
            printf("Error : Unable to compute ThirdReLU backward\n");
            exit(0);
        }
        for(int k = 0;k < depth;k++){
            for(int i = 0;i < height;i++){
                for(int j = 0;j < width;j++){
                    error[n].thiArr[k].arr[i][j] = ReLUDer(output[n].thiArr[k].arr[i][j]);
                }
            }
        }
        if(!error[n].CheckFinite()){
            fprintf(fpDebug, "In ReLuLayer: Array called error\n");
            fflush(fpDebug);
            exit(0);
        }
        error[n] = DotProduct(error[n],err[n]);
        //error[n].Bound(minStep,maxStep);
    }
    return;
}