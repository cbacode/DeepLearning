#include "DeepLearning.h"

//    int inpSize;
//    int outSize;
//    int width;
//    Array coff;
//    Array bias;
//    Array output;
//    Array error;

LinearLayer::LinearLayer(int inp, int out, int w){
    inpSize = inp;
    outSize = out;
    width = w;
    coff.ChangeSize(out,inp);
    bias.ChangeSize(out,w);
    error.ChangeSize(inp,w);
    InitCoff();
    InitBias();
    //output.ChangeSize(out,w);
    //error.ChangeSize(w,inp);
}

void LinearLayer::forward(Array* input){
    inp = (*input);
    output = coff * inp + bias;
    if(!output.CheckFinite()){
        fprintf(fpDebug, "In LinearLayer: Array called output\n");
        fflush(fpDebug);
        exit(0);
    }
    // output.Bound(minStep,maxStep);
    return;
}

void LinearLayer::backward(Array* err){
    //fprintf(fpDebug,"In LinearLayer:\n");
    //err->PrintArray(fpDebug);
    bias = bias + (-LearningRate) * (*err);
    //bias.PrintArray(fpDebug);
    error = coff.Transfer() * (*err);
    coff = coff + (-LearningRate) * (*err) * inp.Transfer();
    if(!bias.CheckFinite()){
        fprintf(fpDebug, "In LinearLayer: Array called bias\n");
        fflush(fpDebug);
        exit(0);
    }
    if(!coff.CheckFinite()){
        fprintf(fpDebug, "In LinearLayer: Array called coff\n");
        fflush(fpDebug);
        exit(0);
    }
    if(!error.CheckFinite()){
        fprintf(fpDebug, "In LinearLayer: Array called error\n");
        fflush(fpDebug);
        exit(0);
    }
    error.Bound(minStep,maxStep);
    //fprintf(fpDebug,"In LinearLayer: \n");
    //error.PrintArray(fpDebug);
    return;
}

void LinearLayer::InitCoff(void){
    default_random_engine e; //engine
	normal_distribution<double> n(0, 1); //(mu,sigma)
    for(int i = 0;i < outSize;i++){
        for(int j = 0;j < inpSize;j++){
            coff.arr[i][j] = n(e);
        }
    }
    return;
}

// Using this initialize method to fasten training
void LinearLayer::InitBias(void){
    default_random_engine e; 
	normal_distribution<double> n(0, 1); 
    for(int i = 0;i < outSize;i++){
        for(int j = 0;j < width;j++){
            bias.arr[i][j] = n(e);
        }    
    }
    return;
}