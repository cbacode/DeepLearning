#include "DeepLearning.h"

//    int height;
//    int width;
//    Array output;
//    Array error;

SigmoidLayer::SigmoidLayer(int h, int w){
    height = h;
    width = w;
    for(int i = 0;i < batchSize;i++){
        Array tmp(height,width);
        output.push_back(tmp);
        error.push_back(tmp);
    }
}

double SigmoidLayer::Sigmoid(double x){
    return 1/(1+exp(-x));
}

void SigmoidLayer::forward(const vector<Array>& inp){
    for(int n = 0;n < batchSize;n++){
        if(inp[n].width != width || inp[n].height != height){
            printf("Error : Unable to compute Sigmoid Forward\n");
            exit(0);
        }
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                output[n].arr[i][j] = Sigmoid(inp[n].arr[i][j]);
            }
        }
        if(!output[n].CheckFinite()){
            fprintf(fpDebug, "In SigmoidLayer: Array called output\n");
            fflush(fpDebug);
            exit(0);
        }
    }
    return;
}

// void SigmoidLayer::backward(int num,vector<Array>* err){
//     vector<Array> inp = (*err);
//     for(int n = 0;n < num;n++){
//         if(inp[n].width != width || inp[n].height != height){
//             printf("Error : Unable to compute Sigmoid backward\n");
//             exit(0);
//         }
//         for(int i = 0;i < height;i++){
//             for(int j = 0;j < width;j++){
//                 double y = output[n].arr[i][j];
//                 error[n].arr[i][j] = y*(1-y);
//             }
//         }
//         error[n] = DotProduct(error[n],inp[n]);
//         if(!error[n].CheckFinite()){
//             fprintf(fpDebug, "In SigmoidLayer: Array called error\n");
//             fflush(fpDebug);
//             exit(0);
//         }
//     }
//     return;
// }

void SigmoidLayer::backward(const vector<Array>& err){
    error = err;
    return;
}