#include "DeepLearning.h"

ThirdSpanLayer::ThirdSpanLayer(int d,int h,int w){
    depth = d;
    height = h;
    width = w;
    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp(depth,height,width);
        error.push_back(tmp);
    }
    for(int i = 0;i < batchSize;i++){
        Array tmp(depth*height*width,1);
        output.push_back(tmp);
    }
}

void ThirdSpanLayer::forward(const vector<ThirdArray>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].depth != depth || inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute ThirdSpan Forward\n");
            fprintf(fpDebug, "Error : Unable to compute ThirdSpan Forward\n");
            fprintf(fpDebug, "depth = %d\n",depth);
            fprintf(fpDebug, "input->depth = %d\n",inp[n].depth);
            fprintf(fpDebug, "height = %d\n",height);
            fprintf(fpDebug, "input->height = %d\n",inp[n].height);
            fprintf(fpDebug, "width = %d\n",width);
            fprintf(fpDebug, "input->width = %d\n",inp[n].width);
            exit(0);
        }
        for(int i = 0;i < depth;i++){
            for(int j = 0;j < height;j++){
                for(int k = 0;k < width;k++){
                    output[n].arr[i * height * width + j * width + k][0] = inp[n].thiArr[i].arr[j][k];
                }
            }
        }
    }
    return;
}

void ThirdSpanLayer::backward(const vector<Array>& err){
    for(int n = 0;n < batchSize;n++){
        if(err[n].width != 1 || err[n].height != depth * width * height){
            printf("Error : Unable to compute Span backward\n");
            exit(0);
        }
        for(int i = 0;i < depth;i++){
            for(int j = 0;j < height;j++){
                for(int k = 0;k < width;k++){
                    error[n].thiArr[i].arr[j][k] = err[n].arr[i * height * width + j *   width + k][0];
                }
            }
        }
    }
    return;
}