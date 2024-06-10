#include "DeepLearning.h"

BatchLayer::BatchLayer(int id,int ih,int iw){
    epoch = 0;
    inpDepth = id;
    inpHeight = ih;
    inpWidth = iw; 
    ThirdArray tmp(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        output.push_back(tmp);
        error.push_back(tmp);
        norm.push_back(tmp);
        singleDiffGamma.push_back(tmp);
        singleDiffSigma.push_back(tmp);
        singleDiffMuLeft.push_back(tmp);
        singleDiffMuRight.push_back(tmp);
    }
    mu.ChangeSize(inpDepth,inpHeight,inpWidth);
    sigma.ChangeSize(inpDepth,inpHeight,inpWidth);
    testMu.ChangeSize(inpDepth,inpHeight,inpWidth);
    testSigma.ChangeSize(inpDepth,inpHeight,inpWidth);
    gamma.ChangeSize(inpDepth,inpHeight,inpWidth);
    beta.ChangeSize(inpDepth,inpHeight,inpWidth);
    momGamma.ChangeSize(inpDepth,inpHeight,inpWidth);
    momBeta.ChangeSize(inpDepth,inpHeight,inpWidth);
    rmsGamma.ChangeSize(inpDepth,inpHeight,inpWidth);
    rmsBeta.ChangeSize(inpDepth,inpHeight,inpWidth);
    diffGamma.ChangeSize(inpDepth,inpHeight,inpWidth);
    diffBeta.ChangeSize(inpDepth,inpHeight,inpWidth);
    InitCoff();
    return;
}

void BatchLayer::InitCoff(void){
    for(int i = 0;i < inpDepth;i++){
        for(int j = 0;j < inpHeight;j++){
            for(int k = 0;k < inpWidth;k++){
                gamma.thiArr[i].arr[j][k] = 1;
                beta.thiArr[i].arr[j][k] = 0;
            }
        }
    }
    return;
}

void BatchLayer::forward(const vector<ThirdArray>& inp){
    mu = 0;
    sigma = 0;
    for(int i = 0;i < batchSize;i++){
        mu = mu + (1.0 / batchSize) * inp[i];
    }
    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp = inp[i] - mu;
        sigma = sigma + (1.0 / batchSize) * DotProduct(tmp, tmp);
    }
    //testSigma = momCoff * testSigma + (1 - momCoff) * sigma;
    //testMu = momCoff * testMu + (1 - momCoff) * mu;
    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&BatchLayer::ThreadForward, this, i, inp[i]);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    for(int i = 0;i < batchSize;i++){
        if(!output[i].CheckFinite()){
            fprintf(fpDebug,"In BatchLayer : %d\n",i);
            exit(0);
        }
    }
    return;
}

void BatchLayer::testForward(const vector<ThirdArray>& inp){
    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&BatchLayer::ThreadTestForward, this, i, inp[i]);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    for(int i = 0;i < batchSize;i++){
        if(!output[i].CheckFinite()){
            fprintf(fpDebug,"In BatchLayer : %d\n",i);
            exit(0);
        }
    }
    return;
}

void BatchLayer::ThreadForward(int i,ThirdArray inp){
    norm[i] = (inp - mu) / sqrt(sigma + eps);
    output[i] = DotProduct(norm[i], gamma) + beta;
}

void BatchLayer::ThreadTestForward(int i,ThirdArray inp){
    norm[i] = (inp - testMu) / sqrt(testSigma + eps);
    output[i] = DotProduct(norm[i], gamma) + beta;
}

void BatchLayer::backward(const vector<ThirdArray>& err){
    ThirdArray diffSigma(inpDepth,inpHeight,inpWidth);
    ThirdArray diffMu(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        thr[i] = thread(&BatchLayer::ThreadBackward, this, i, err[i]);
    }
    for(int i = 0;i < batchSize;i++){
        thr[i].join();
    }
    diffBeta = 0;
    diffGamma = 0;
    for(int i = 0;i < batchSize;i++){
        diffBeta = diffBeta + err[i];
        diffGamma = diffGamma + singleDiffGamma[i];
        diffSigma = diffSigma - singleDiffSigma[i];
    }
    
    for(int i = 0;i < batchSize;i++){
        diffMu = diffMu - singleDiffMuLeft[i] - DotProduct(diffSigma, singleDiffMuRight[i]);
    }
    for(int i = 0;i < batchSize;i++){
        error[i] = singleDiffMuLeft[i] + DotProduct(diffSigma, singleDiffMuRight[i]) + (1.0 / batchSize) * diffMu;
    }
    epoch = epoch + 1;
    diffBeta = (1.0 / batchSize) * diffBeta;
    diffGamma = (1.0 / batchSize) * diffGamma;
    //Adam(gamma,momGamma,rmsGamma,diffGamma,epoch);
    //Adam(beta,momBeta,rmsBeta,diffBeta,epoch);

    for(int i = 0;i < batchSize;i++){
        if(!error[i].CheckFinite()){
            fprintf(fpDebug,"In BatchLayer : %d\n",i);
            exit(0);
        }
    }
    return;
}

void BatchLayer::ThreadBackward(int i,const ThirdArray& err){
    singleDiffGamma[i] = DotProduct(err, norm[i]); 
    ThirdArray diffNorm = DotProduct((1.0 / batchSize) * err, gamma);
    // 1 / batchSize
    ThirdArray diffSigma = DotProduct(diffNorm, norm[i]);
    // 1 / batchSize * norm[i]
    ThirdArray sqrtSigma = sqrt(sigma + eps);
    singleDiffSigma[i] = (1.0 / 2.0) * diffSigma / (sigma + eps) ;
    singleDiffMuLeft[i] = diffNorm / sqrtSigma;
    // diffMuRight = inp[i] - mu;
    singleDiffMuRight[i] = (2.0 / batchSize) * DotProduct(norm[i], sqrtSigma);
    return;
}

void BatchLayer::Adam(ThirdArray &var,ThirdArray& cacheMom,ThirdArray& cacheRms,ThirdArray& diff,int epoch){
    cacheMom = momCoff * cacheMom + (1.0 - momCoff) * diff;
    cacheRms = rmsCoff * cacheRms + (1.0 - rmsCoff) * DotProduct(diff , diff);
    ThirdArray momCorr = 1.0/(1 - pow(momCoff,epoch)) * cacheMom;
    ThirdArray rmsCorr = 1.0/(1 - pow(rmsCoff,epoch)) * cacheRms;
    ThirdArray diffVar = momCorr / ( sqrt(rmsCorr + eps));
    var = var - LearningRate * (diffVar + regular * var); 
    return;   
}