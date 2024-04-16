#include "DeepLearning.h"

SpanLayer::SpanLayer(int h, int w){
    height = h;
    width = w;
    output.ChangeSize(h*w,1);
    error.ChangeSize(h,w);
}

void SpanLayer::forward(Array* input){
    if(input->width != width || input->height != height){
        printf("Error : Unable to compute Span Forward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            output.arr[i * width + j][0] = input->arr[i][j];
        }
    }
    return;
}

void SpanLayer::backward(Array* err){
    if(err->width != 1 || err->height != width * height){
        printf("Error : Unable to compute Span backward\n");
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            error.arr[i][j] = err->arr[i * width + j][0];
        }
    }
    return;
}