#include "DeepLearning.h"

SoftMaxLayer::SoftMaxLayer(int h, int w){
    height = h;
    width = w;
    sum.resize(width);
    for(int i = 0;i < batchSize;i++){
        Array tmp(height,width);
        output.push_back(tmp);
        error.push_back(tmp);
    }
}

void SoftMaxLayer::AddTogether(int n){
    for(int j = 0;j < width;j++){
        sum[j] = 0;
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            sum[j] += output[n].arr[i][j];
        }
    }
    for(int j = 0;j < width;j++){
        if(abs(sum[j]) < 1e-5){
            int sign = (sum[j] > 0) ? 1 : -1;
            sum[j] = 1e-5 * sign;
        }
    }
}

void SoftMaxLayer::forward(const vector<Array>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute SoftMax Forward\n");
            exit(0);
        }
        double mini = 1e10;
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                if(inp[n].arr[i][j] < mini){
                    mini = inp[n].arr[i][j];
                }
            }
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output[n].arr[i][j] = exp(inp[n].arr[i][j] - mini);
            }
        }
        AddTogether(n);
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output[n].arr[i][j] = output[n].arr[i][j]/sum[j];
            }
        }
        if(!output[n].CheckFinite()){
            fprintf(fpDebug, "In SoftMaxLayer: Array called output\n");
            fflush(fpDebug);
            exit(0);
        }
    }
    return;
}

void SoftMaxLayer::backward(const vector<Array>& err){
    error = err;
    return;
}