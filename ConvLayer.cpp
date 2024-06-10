#include "DeepLearning.h"

ConvLayer::ConvLayer(int id,int ih,int iw,int kd,int kh,int kw,int num,int s,int pad){
    epoch = 0;
    inpDepth = id;
    inpHeight = ih;
    inpWidth = iw;

    kerDepth = kd;
    kerHeight = kh;
    kerWidth = kw;
    if(inpDepth != kerDepth){
        fprintf(fpDebug,"Bad kerDepth chosen in ConvLayer.\n");
        exit(0);
    }

    numKernel = num;
    stride = s;
    padding = pad;

    outDepth = num;
    outHeight = (inpHeight + 2 * padding - kerHeight) / stride + 1;
    outWidth = (inpWidth + 2 * padding - kerWidth) / stride + 1;
    
    for(int i = 0;i < numKernel;i++){
        ThirdArray arr(kerDepth,kerHeight,kerWidth);
        kernal.push_back(arr);
        momForKernal.push_back(arr);
        rmsForKernal.push_back(arr);
    }
    for(int i = 0;i < batchSize;i++){
        ThirdArray arr(inpDepth,inpHeight,inpWidth);
        input.push_back(arr);
    }
    for(int i = 0;i < batchSize;i++){
        ThirdArray arr(outDepth,outHeight,outWidth);
        output.push_back(arr);
    }
    for(int i = 0;i < batchSize;i++){
        ThirdArray arr(inpDepth,inpHeight,inpWidth);
        error.push_back(arr);
    }
    bias.resize(outDepth);
    momForBias.resize(outDepth);
    rmsForBias.resize(outDepth);

    for(int i = 0;i < numKernel;i++){
        ThirdArray tmp(kerDepth,kerHeight,kerWidth);
        diffKer.push_back(tmp);
    }
    diffBias.resize(numKernel);
    InitKernel();
}

void ConvLayer::InitKernel(void){
    int inpSize = kerDepth * kerHeight * kerWidth;
    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    double bound = sqrt(2.0 / inpSize);
    for(int i = 0;i < numKernel;i++){
        for(int j = 0;j < kerDepth;j++){
            for(int m = 0;m < kerHeight;m++){
                for(int n = 0;n < kerWidth;n++){
                    kernal[i].thiArr[j].arr[m][n] = bound * nor(e);
                }    
            }
        }
        bias[i] = 0;
    }
    return;
}

void ConvLayer::forward(const vector<ThirdArray>& inp){
    input = inp;
    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&ConvLayer::threadForward, this, i);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    for(int i = 0;i < batchSize;i++){
        if(!output[i].CheckFinite()){
            fprintf(fpDebug,"In ConvLayer : %d\n",i);
            exit(0);
        }
    }
    return;
}

void ConvLayer::threadForward(int i){
    for(int j = 0;j < numKernel;j++){
        output[i].thiArr[j] = ThirdConv(input[i], kernal[j], padding, stride) + bias[j];
    }
    return;
}

void ConvLayer::backward(const vector<ThirdArray>& err){
    vector<vector<ThirdArray>> SingleDiffKer;
    vector<vector<double>> SingleDiffBias;

    for(int j = 0;j < numKernel;j++){
        diffBias[j] = 0;
        diffKer[j] = 0;
    }
    for(int i = 0;i < batchSize;i++){
        SingleDiffBias.push_back(diffBias);
        SingleDiffKer.push_back(diffKer);
    }

    for(int i = 0;i < batchSize;i++){
        error[i] = 0;
        thr[i] = thread(&ConvLayer::threadBackward, this, ref(SingleDiffBias[i]), ref(SingleDiffKer[i]), ref(error[i]), err[i], input[i]);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }

    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < numKernel;j++){
            diffBias[j] = diffBias[j] + SingleDiffBias[i][j];
            diffKer[j] = diffKer[j] + SingleDiffKer[i][j];
        }
        //error[i].PrintArray(fpDebug);  
    }
    epoch = epoch + 1;
    for(int j = 0;j < numKernel;j++){
        if(adam){
            Adam(bias[j],momForBias[j],rmsForBias[j],diffBias[j],epoch);
            Adam(kernal[j],momForKernal[j],rmsForKernal[j],diffKer[j],epoch);
        }
        else{
            bias[j] = bias[j] - LearningRate * (diffBias[j] + regular * bias[j]); 
            kernal[j] = kernal[j] - LearningRate * (diffKer[j] + regular * kernal[j]); 
        }
        
    }

    for(int i = 0;i < batchSize;i++){
        if(!error[i].CheckFinite()){
            fprintf(fpDebug,"In ConvLayer : %d\n",i);
            exit(0);
        }
    }
    return;
}

void ConvLayer::threadBackward(vector<double>& DiffBias,vector<ThirdArray>& DiffKer,ThirdArray& error,ThirdArray err,ThirdArray inp){
    int errKerHeight = inpHeight + 2 * padding - kerHeight + 1;
    int errKerWidth = inpWidth + 2 * padding - kerWidth + 1;
    Array tmp(outHeight,outWidth);
    for(int j = 0;j < numKernel;j++){
        for(int k = 0;k < kerDepth;k++){
            tmp = err.thiArr[j].Spread(stride,errKerHeight,errKerWidth);
            Array diffKernel = Conv(inp.thiArr[k],tmp,padding,1);
            DiffKer[j].thiArr[k] = (1.0 / batchSize) * diffKernel;
            int pad = (errKerHeight + kerHeight - err.height - 1) / 2;
            Array ker = kernal[j].thiArr[k].Rotate();
            error.thiArr[k] = error.thiArr[k] + Conv(tmp,ker,pad,1);
        }
        DiffBias[j] = (1.0 / batchSize) * err.thiArr[j].addTogether();   
    }
    return;
}

void ConvLayer::Adam(double &var,double& cacheMom,double& cacheRms,double& diff,int epoch){
    cacheMom = momCoff * cacheMom + (1.0 - momCoff) * diff;
    cacheRms = rmsCoff * cacheRms + (1.0 - rmsCoff) * diff * diff;
    double momCorr = 1.0/(1 - pow(momCoff,epoch)) * cacheMom;
    double rmsCorr = 1.0/(1 - pow(rmsCoff,epoch)) * cacheRms;
    double diffVar = momCorr / ( sqrt(rmsCorr + eps));
    var = var - LearningRate * (diffVar + regular * var); 
    return;   
}

void ConvLayer::Adam(ThirdArray &var,ThirdArray& cacheMom,ThirdArray& cacheRms,ThirdArray& diff,int epoch){
    cacheMom = momCoff * cacheMom + (1.0 - momCoff) * diff;
    cacheRms = rmsCoff * cacheRms + (1.0 - rmsCoff) * DotProduct(diff , diff);
    ThirdArray momCorr = 1.0/(1 - pow(momCoff,epoch)) * cacheMom;
    ThirdArray rmsCorr = 1.0/(1 - pow(rmsCoff,epoch)) * cacheRms;
    ThirdArray diffVar = momCorr / ( sqrt(rmsCorr + eps));
    var = var - LearningRate * (diffVar + regular * var); 
    return;   
}