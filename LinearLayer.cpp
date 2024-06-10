#include "DeepLearning.h"

//    int inpSize;
//    int outSize;
//    int width;
//    Array coff;
//    Array bias;
//    Array output;
//    Array error;

LinearLayer::LinearLayer(int inp, int out){
    epoch = 0;
    inpHeight = inp;
    inpWidth = 1;
    outHeight = out;
    outWidth = 1;
    coff.ChangeSize(outHeight,inpHeight);
    bias.ChangeSize(outHeight,outWidth);
    diffBias.ChangeSize(outHeight,outWidth);
    diffCoff.ChangeSize(outHeight,inpHeight);
    for(int i = 0;i < batchSize;i++){
        Array tmp(inpHeight,inpWidth);
        error.push_back(tmp);
    }
    for(int i = 0;i < batchSize;i++){
        Array tmp(outHeight,outWidth);
        output.push_back(tmp);
    }
    momForCoff.ChangeSize(outHeight,inpHeight);
    rmsForCoff.ChangeSize(outHeight,inpHeight);
    momForBias.ChangeSize(outHeight,outWidth);
    rmsForBias.ChangeSize(outHeight,outWidth);
    InitCoff();
    InitBias();
}

void LinearLayer::forward(const vector<Array>& input){
    inp = input;
    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&LinearLayer::threadForward, this, i);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    for(int i = 0;i < batchSize;i++){
        if(!output[i].CheckFinite()){
            fprintf(fpDebug, "In SoftMaxLayer: Array called output\n");
            fflush(fpDebug);
            exit(0);
        }
    }
    return;
}

void LinearLayer::threadForward(int i){
    output[i] = coff * inp[i] + bias;
    if(!output[i].CheckFinite()){
        fprintf(fpDebug, "In LinearLayer: Array called output\n");
        fflush(fpDebug);
        exit(0);
    }
    //output[i].Bound(minStep,maxStep);
}

void LinearLayer::backward(const vector<Array>& err){
    vector<Array> SingleDiffBias;
    vector<Array> SingleDiffCoff;

    diffBias = 0;
    diffCoff = 0;

    for(int i = 0;i < batchSize;i++){
        SingleDiffBias.push_back(diffBias);
        SingleDiffCoff.push_back(diffCoff);
    }

    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&LinearLayer::threadBackward, this, i, ref(SingleDiffBias[i]), ref(SingleDiffCoff[i]), err[i]);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    for(int i = 0;i < batchSize;i++){
        diffBias = diffBias + SingleDiffBias[i];
        diffCoff = diffCoff + SingleDiffCoff[i];
    }
    epoch = epoch + 1;
    if(adam){
        Adam(bias,momForBias,rmsForBias,diffBias,epoch);
        Adam(coff,momForCoff,rmsForCoff,diffCoff,epoch);
    }
    else{
        bias = bias - LearningRate * (diffBias + regular * bias); 
        coff = coff - LearningRate * (diffCoff + regular * coff); 
    }
    

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
    for(int i = 0;i < batchSize;i++){
        if(!error[i].CheckFinite()){
            fprintf(fpDebug, "In LinearLayer: Array called error\n");
            fflush(fpDebug);
            exit(0);
        }
        //error[i].Bound(minStep,maxStep);
    }
    return;
}

void LinearLayer::threadBackward(int i,Array& DiffBias,Array& diffCoff,const Array& err){
    DiffBias = (1.0 / batchSize) * err;
    diffCoff = (1.0 / batchSize) * err * inp[i].Transfer();
    error[i] = coff.Transfer() * err;
    //error[i].PrintArray(fpDebug);
    return;
}

// Using this initialize method to fasten training
void LinearLayer::InitCoff(void){
    default_random_engine e; //engine
	normal_distribution<double> n(0, 1); //(mu,sigma)
    int inpSize = inpHeight * inpWidth;
    int outSize = outHeight * outWidth;
    double bound = sqrt(2.0 / inpSize);
    for(int i = 0;i < outHeight;i++){
        //int ii = i % 49;
        for(int j = 0;j < inpHeight;j++){
            int jj = j % 49;
            coff.arr[i][j] = n(e) * bound;
        }
    }
    return;
}

// Using this initialize method to fasten training
void LinearLayer::InitBias(void){
    default_random_engine e; 
	normal_distribution<double> n(0, 1); 
    int inpSize = inpHeight * inpWidth;
    int outSize = outHeight * outWidth;
    double bound = sqrt(2.0 / inpSize);
    for(int i = 0;i < outHeight;i++){
        for(int j = 0;j < outWidth;j++){
            //bias.arr[i][j] = n(e) * bound;
            bias.arr[i][j] = 0;
        }    
    }
    return;
}

void LinearLayer::Adam(Array &var,Array& cacheMom,Array& cacheRms,Array& diff,int epoch){
    cacheMom = momCoff * cacheMom + (1.0 - momCoff) * diff;
    cacheRms = rmsCoff * cacheRms + (1.0 - rmsCoff) * DotProduct(diff , diff);
    Array momCorr = 1.0/(1 - pow(momCoff,epoch)) * cacheMom;
    Array rmsCorr = 1.0/(1 - pow(rmsCoff,epoch)) * cacheRms;
    Array diffVar = momCorr / ( sqrt(rmsCorr + eps));
    var = var - LearningRate * (diffVar + regular * var); 
    return;   
}