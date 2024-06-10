#include "DeepLearning.h"

MaxPoolingLayer::MaxPoolingLayer(int id,int ih,int iw,int kd,int kh,int kw,int st,int ty){
    inpDepth = id;
    inpHeight = ih;
    inpWidth = iw;

    kerDepth = kd;
    kerHeight = kh;
    kerWidth = kw;

    if(inpDepth != kerDepth){
        fprintf(fpDebug,"Bad inpDepth or kerDepth chosen in MaxPoolingLayer.\n");
        exit(0);
    }

    stride = st;
    type = ty;

    if(type != 1 && type != 2){
        fprintf(fpDebug,"Bad type chosen in MaxPoolingLayer.\n");
        exit(0);
    }

    if((inpHeight - kerHeight) % stride != 0){
        fprintf(fpDebug,"Bad kerHeight or stride chosen in MaxPoolingLayer.\n");
        exit(0);
    }
    if((inpWidth - kerWidth) % stride != 0){
        fprintf(fpDebug,"Bad kerWidth or stride chosen in MaxPoolingLayer.\n");
        exit(0);
    }

    outDepth = inpDepth;
    outHeight = (inpHeight - kerHeight) / stride + 1;
    outWidth = (inpWidth - kerWidth) / stride + 1;

    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp(outDepth, outHeight, outWidth);
        output.push_back(tmp);
        place.push_back(tmp);
    }
    for(int i = 0;i < batchSize;i++){
        ThirdArray tmp(inpDepth, inpHeight, inpWidth);
        error.push_back(tmp);
    }
    return;
}

void MaxPoolingLayer::forward(const vector<ThirdArray>& inp){
    for(int n = 0;n < batchSize;n++){
        output[n] = 0;
        place[n] = 0;
        for(int i = 0;i < outDepth;i++){
            for(int j = 0;j < outHeight;j++){
                for(int k = 0;k < outWidth;k++){
                    if(type == 1){
                        MaxPoolingForward(n,inp[n],i,j,k);
                    }
                    else{
                        AvgPoolingForward(n,inp[n],i,j,k);
                    }
                }
            }    
        }
        //inp[n].PrintArray(fpDebug);
        //place[n].PrintArray(fpDebug);
    }
    return;
}

void MaxPoolingLayer::MaxPoolingForward(int tar,const ThirdArray& inp,int d,int h,int w){
    int x = 0;
    int y = 0;
    output[tar].thiArr[d].arr[h][w] = - maxStep;
    for(int i = 0;i < kerHeight;i++){
        for(int j = 0;j < kerWidth;j++){
            x = h * stride + i;
            y = w * stride + j;
            if(x < 0 || x >= inpHeight){
                printf("Unable to compute MaxPoolingForward.\n");
                fprintf(fpDebug, "Unable to compute MaxPoolingForward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            if(y < 0 || y >= inpWidth){
                printf("Unable to compute MaxPoolingForward.\n");
                fprintf(fpDebug, "Unable to compute MaxPoolingForward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            if(output[tar].thiArr[d].arr[h][w] < inp.thiArr[d].arr[x][y]){
                output[tar].thiArr[d].arr[h][w] = inp.thiArr[d].arr[x][y];
                place[tar].thiArr[d].arr[h][w] = i * kerWidth + j;
            }
        }
    }
    return;
}

void MaxPoolingLayer::AvgPoolingForward(int tar,const ThirdArray& inp,int d,int h,int w){
    int coff = kerHeight * kerWidth;
    for(int i = 0;i < kerHeight;i++){
        for(int j = 0;j < kerWidth;j++){
            int x = h * stride + i;
            int y = w * stride + j;
            if(x < 0 || x >= inpHeight){
                printf("Unable to compute AvgPoolingForward.\n");
                fprintf(fpDebug, "Unable to compute AvgPoolingForward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            if(y < 0 || y >= inpWidth){
                printf("Unable to compute AvgPoolingForward.\n");
                fprintf(fpDebug, "Unable to compute AvgPoolingForward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            output[tar].thiArr[d].arr[h][w] += (inp.thiArr[d].arr[x][y] / coff);
        }
    }
    return;
}

void MaxPoolingLayer::backward(const vector<ThirdArray>& err){
    for(int n = 0;n < batchSize;n++){
        error[n] = 0;
        for(int i = 0;i < inpDepth;i++){
            for(int j = 0;j < outHeight;j++){
                for(int k = 0;k < outWidth;k++){
                    if(type == 1){
                        MaxPoolingBackward(n,err[n],i,j,k);
                    }
                    else{
                        AvgPoolingBackward(n,err[n],i,j,k);
                    }
                }
            }
        }
    }
    return;
}

void MaxPoolingLayer::MaxPoolingBackward(int tar,const ThirdArray& err,int d,int h,int w){
    int dx = place[tar].thiArr[d].arr[h][w] / kerWidth;
    int dy = (int)place[tar].thiArr[d].arr[h][w] % kerWidth;
    int x = h * stride + dx;
    int y = w * stride + dy;
    if(x < 0 || x >= inpHeight){
        printf("Unable to compute MaxPoolingBackward.\n");
        fprintf(fpDebug, "Unable to compute MaxPoolingBackward.\n");
        fprintf(fpDebug, "dx = %d, dy = %d\n",dx,dy);
        fprintf(fpDebug, "x = %d, y = %d\n",x,y);
        fflush(fpDebug);
        exit(0);
    }
    if(y < 0 || y >= inpWidth){
        printf("Unable to compute MaxPoolingBackward.\n");
        fprintf(fpDebug, "Unable to compute MaxPoolingBackward.\n");
        fprintf(fpDebug, "dx = %d, dy = %d\n",dx,dy);
        fprintf(fpDebug, "x = %d, y = %d\n",x,y);
        fflush(fpDebug);
        exit(0);
    }
    error[tar].thiArr[d].arr[x][y] = err.thiArr[d].arr[h][w];
    return;
}

void MaxPoolingLayer::AvgPoolingBackward(int tar,const ThirdArray& err,int d,int h,int w){
    int coff = kerHeight * kerWidth;
    for(int i = 0;i < kerHeight;i++){
        for(int j = 0;j < kerWidth;j++){
            int x = h * stride + i;
            int y = w * stride + j;
            if(x < 0 || x >= inpHeight){
                printf("Unable to compute AvgPoolingBackward.\n");
                fprintf(fpDebug, "Unable to compute AvgPoolingBackward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            if(y < 0 || y >= inpWidth){
                printf("Unable to compute AvgPoolingBackward.\n");
                fprintf(fpDebug, "Unable to compute AvgPoolingBackward.\n");
                fprintf(fpDebug, "i = %d, j = %d\n",i,j);
                fprintf(fpDebug, "x = %d, y = %d\n",x,y);
                fflush(fpDebug);
                exit(0);
            }
            error[tar].thiArr[d].arr[x][y] += (err.thiArr[d].arr[h][w] / coff);
        }
    }
    return;
}