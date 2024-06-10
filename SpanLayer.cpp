#include "DeepLearning.h"

SpanLayer::SpanLayer(int h, int w){
    height = h;
    width = w;
    for(int i = 0;i < batchSize;i++){
        Array tmp(height,width);
        error.push_back(tmp);
    }
    for(int i = 0;i < batchSize;i++){
        Array tmp(height*width,1);
        output.push_back(tmp);
    }
}

void SpanLayer::forward(const vector<Array>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute Span Forward\n");
            exit(0);
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output[n].arr[i * width + j][0] = inp[n].arr[i][j];
            }
        }
    }
    return;
}

void SpanLayer::backward(const vector<Array>& err){
    for(int n = 0;n < batchSize;n++){
        if(err[n].width != 1 || err[n].height != width * height){
            printf("Error : Unable to compute Span backward\n");
            exit(0);
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                error[n].arr[i][j] = err[n].arr[i * width + j][0];
            }
        }
    }
    return;
}