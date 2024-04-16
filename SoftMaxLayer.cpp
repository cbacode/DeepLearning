#include "DeepLearning.h"

SoftMaxLayer::SoftMaxLayer(int h, int w){
    height = h;
    width = w;
    //width = 1;
    sum.resize(width);
    //input.ChangeSize(height,width);
    output.ChangeSize(height,width);
    error.ChangeSize(height,width);
}

void SoftMaxLayer::AddTogether(void){
    for(int j = 0;j < width;j++){
        sum[j] = 0;
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            sum[j] += output.arr[i][j];
        }
    }
    for(int j = 0;j < width;j++){
        if(abs(sum[j]) < 1e-5){
            int sign = (sum[j] > 0) ? 1 : -1;
            sum[j] = 1e-5 * sign;
        }
    }
}

void SoftMaxLayer::SoftMaxDer(int h,int w,Array *err){
    //fprintf(fpDebug,"SoftMax.sum[%d] = : %llf \n",w,sum[w]);
    //double expNum = exp(input.arr[h][w]);
    //error.arr[h][w] = (expNum)*(sum[w]-expNum)/pow(sum[w],2);

    // double num = output.arr[h][w];
    // error.arr[h][w] += num * (1 - num) * err->arr[h][w];
    // for(int i = 0;i < height;i++){
    //     if(i != h){
    //         error.arr[i][w] -= num * output.arr[i][w] * err->arr[h][w];
    //     } 
    // }
}

void SoftMaxLayer::forward(Array* inp){
    input = (*inp);
    //fprintf(fpDebug, "SoftMax.input = \n");
    //input.PrintArray(fpDebug);
    if(input.width != width || input.height != height){
        printf("Error : Unable to compute SoftMax Forward\n");
        exit(0);
    }
    input.Bound(minStep,maxInput);
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            output.arr[i][j] = exp(input.arr[i][j]);
        }
    }
    //fprintf(fpDebug, "SoftMax.output = \n");
    //output.PrintArray(fpDebug);
    AddTogether();
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            output.arr[i][j] = output.arr[i][j]/sum[j];
        }
    }
    if(!output.CheckFinite()){
        input.PrintArray(fpDebug);
        fprintf(fpDebug, "In SoftMaxLayer: Array called output\n");
        fflush(fpDebug);
        exit(0);
    }
    //fprintf(fpDebug, "SoftMax.output = \n");
    //output.PrintArray(fpDebug);
    return;
}

void SoftMaxLayer::backward(Array* err){
    if(err->width != width || err->height != height){
        printf("Error : Unable to compute SoftMax backward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            error.arr[i][j] = 0;
        }
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            SoftMaxDer(i,j,err);
        }
    }
    if(!error.CheckFinite()){
        fprintf(fpDebug, "In SoftMaxLayer: Array called error\n");
        fflush(fpDebug);
        exit(0);
    }
    //error = DotProduct(error,(*err));
    //error = LearningRate * error;
    error = *err;
    // fprintf(fpDebug,"In SoftMaxLayer.error: \n");
    // error.PrintArray(fpDebug);
    error.Bound(minStep,maxStep);
    return;
}