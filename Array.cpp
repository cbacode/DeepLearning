#include "DeepLearning.h"

//    vector<vector<double>> arr;
//    int width;
//    int height;

Array::Array(int hei,int wid){
    width = wid;
    height = hei;
    for(int i = 0;i < height;i++){
        vector<double> temp(width,0);
        arr.push_back(temp);
    }
}

Array::Array(void){
    width = 0;
    height = 0;
}

void Array::PrintArray(FILE* fpDebug){
    fprintf(fpDebug,"%d %d\n",arr.size(),arr[0].size());
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            fprintf(fpDebug,"%llf ",arr[i][j]);
        }
        fprintf(fpDebug,";\n");
    }
    fprintf(fpDebug,"\n");
    fflush(fpDebug);
    return;
}

void Array::PrintMatlabArray(FILE* valid){
    fprintf(valid,"[ ");
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            fprintf(valid,"%llf ",arr[i][j]);
        }
        fprintf(valid,";\n");
    }
    fprintf(valid,"];\n");
    fflush(valid);
    return;
}

void Array::PrintArrayAvg(FILE* fpDebug){
    fprintf(fpDebug,"%d %d\n",arr.size(),arr[0].size());
    double avg = 0;
    double sqr = 0;
    double maxinum = -1e10;
    double mininum = 1e10;
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            if(arr[i][j] > maxinum){
                maxinum = arr[i][j];
            }
            if(arr[i][j] < mininum){
                mininum = arr[i][j];
            }
            avg += arr[i][j];
        }
    }
    avg /= (height * width);
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            sqr += (arr[i][j] - avg) * (arr[i][j] - avg);
        }
    }
    sqr /= (height * width);
    fprintf(fpDebug,"avg = %llf, sqr = %llf, maxinum = %llf, mininum = %llf\n",avg,sqr,maxinum,mininum);
    fflush(fpDebug);
    return;
}

void Array::ChangeSize(int h,int w){
    width = w;
    height = h;
    for(int i = 0;i < height;i++){
        vector<double> temp(width,0);
        arr.push_back(temp);
    }
    return;
}

void Array::operator=(Array right){
    if(width == 0 && height == 0){
        ChangeSize(right.height,right.width);
    }
    else if(width !=right.width || height != right.height){
        printf("Error : Unable to equal matrix\n");
        fprintf(fpDebug, "Error : Unable to equal matrix\n");
        fprintf(fpDebug, "left.height = %d\n",height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            arr[i][j] = right.arr[i][j];
        }
    }
    return;
}

void Array::operator=(double right){
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            arr[i][j] = right;
        }
    }
    return;
}

Array Array::Spread(int stride,int outHeight,int outWidth){
    Array res(outHeight,outWidth);
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            if(i * stride >= outHeight || j * stride >= outWidth){
                printf("Error in Spread.\n");
                fprintf(fpDebug,"Error in Spread.\n");
                fprintf(fpDebug,"i * stride = %d\n",i * stride);
                fprintf(fpDebug,"j * stride = %d\n",j * stride);
                fprintf(fpDebug,"outHeight = %d\n",outHeight);
                fprintf(fpDebug,"outWidth = %d\n",outWidth);
                fflush(fpDebug);
                exit(0);
            }
            res.arr[i * stride][j * stride] = arr[i][j];
        }
    }
    return res;
}

Array Array::Transfer(void){
    Array res(width,height);
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            res.arr[j][i] = arr[i][j];
        }
    }
    return res;
}

Array Array::Rotate(void){
    Array res(height,width);
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            res.arr[i][j] = arr[height - i - 1][width - j - 1];
        }
    }
    return res;
}

double Array::addTogether(void){
    double res = 0;
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            res += arr[i][j];
        }
    }
    return res;
}

Array Conv(Array inp,Array ker,int pad,int stride){
    int len_hei = (inp.height + 2 * pad - ker.height) / stride + 1;
    int len_wid = (inp.width + 2 * pad - ker.width) / stride + 1;
    //fprintf(fpDebug,"len_hei = %d\n",len_hei);
    //fprintf(fpDebug,"len_wid = %d\n",len_wid);
    //fprintf(fpDebug,"inp_hei = %d\n",inp.height);
    //fprintf(fpDebug,"inp_wid = %d\n",inp.width);
    Array res(len_hei,len_wid);
    for(int i = 0;i < len_hei;i++){
        for(int j = 0;j < len_wid;j++){
            res.arr[i][j] = 0;
            for(int dx = 0;dx < ker.height;dx++){
                for(int dy = 0;dy < ker.width;dy++){
                    int x = i * stride + dx - pad;
                    int y = j * stride + dy - pad;
                    double num = 0;
                    if(x >= 0 && x < inp.height){
                        if(y >= 0 && y < inp.width){
                            num = inp.arr[x][y];
                        }
                    }
                    if(x < -pad || x >= inp.height + pad){
                        fprintf(fpDebug, "Bad padding or stride chosen.\n");
                        exit(0);
                    }
                    if(y < -pad || y >= inp.height + pad){
                        fprintf(fpDebug, "Bad padding or stride chosen.\n");
                        exit(0);
                    }
                    res.arr[i][j] += num * ker.arr[dx][dy];
                }
            }
        }    
    }
    return res;
}

Array DotProduct(Array left,Array right){
    if(left.width != right.width || left.height != right.height){
        printf("Error: Unable to compute DotProduct\n");
        exit(0);
    }
    
    Array res(left.height,left.width);
    for(int h = 0;h < left.height;h++){
        for(int i = 0;i < left.width;i++){
            res.arr[h][i] = left.arr[h][i] * right.arr[h][i];
        }   
    }
    return res;
}

Array sqrt(Array inp){
    Array res(inp.height,inp.width);
    for(int h = 0;h < inp.height;h++){
        for(int i = 0;i < inp.width;i++){
            res.arr[h][i] = sqrt(inp.arr[h][i]);
        }   
    }
    return res;
}

void Array::Bound(double mini, double maxi){
    for(int h = 0;h < height;h++){
        for(int w = 0;w < width;w++){
            if(abs(arr[h][w]) < mini){
                int sign = (arr[h][w] > 0) ? 1 : -1;
                arr[h][w] = mini * sign;
                //fprintf(fpDebug,"Warning : Too small error occur.\n");
                //this->PrintArray(fpDebug);
                //fprintf(fpResult,"Warning : Too small error occur.\n");
            }
            else if(abs(arr[h][w]) > maxi){
                int sign = (arr[h][w] > 0) ? 1 : -1;
                arr[h][w] = maxi * sign;
                //fprintf(fpDebug,"Warning : Too big error occur.\n");
                //this->PrintArray(fpDebug);
                //fprintf(fpResult,"Warning : Too big error occur.\n");
            }
        }
    }
    return;
}

bool Array::CheckFinite(void){
    for(int i = 0;i < height;i++){
        for(int j = 0;j < width;j++){
            if(!isfinite(arr[i][j])){
                fprintf(fpDebug, "Error : Inf or Nan occur in array.\n");
                this->PrintArray(fpDebug);
                printf("Error : Inf or Nan occur in array.\n");
                return false;
                //exit(0);
            }
        }
    }
    return true;
}

Array operator +(Array left, Array right){
    if(left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute matrix adding.\n");
        fprintf(fpDebug, "Error : Unable to compute matrix adding.\n");
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    Array res(left.height,left.width);
    for(int i = 0;i < left.height;i++){
        for(int j = 0;j < left.width;j++){
            res.arr[i][j] = left.arr[i][j] + right.arr[i][j];
        }
    }
    return res;
}

Array operator -(Array left, Array right){
    if(left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute matrix adding.\n");
        fprintf(fpDebug, "Error : Unable to compute matrix adding.\n");
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    Array res(left.height,left.width);
    for(int i = 0;i < left.height;i++){
        for(int j = 0;j < left.width;j++){
            res.arr[i][j] = left.arr[i][j] - right.arr[i][j];
        }
    }
    return res;
}

Array operator /(Array left, Array right){
    if(left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute matrix each divide.\n");
        fprintf(fpDebug, "Error : Unable to compute matrix each divide.\n");
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    Array res(left.height,left.width);
    for(int i = 0;i < left.height;i++){
        for(int j = 0;j < left.width;j++){
            if(abs(right.arr[i][j]) < eps){
                fprintf(fpDebug, "Warning : Trying to divide 0.\n");
                int sign = (right.arr[i][j] > 0) ? 1 : -1;
                right.arr[i][j] = sign * eps;
            }
            res.arr[i][j] = left.arr[i][j] / right.arr[i][j];
        }
    }
    return res;
}

Array operator +(Array left, double num){
    Array res(left.height,left.width);
    for(int i = 0;i < left.height;i++){
        for(int j = 0;j < left.width;j++){
            res.arr[i][j] = left.arr[i][j] + num;
        }
    }
    return res;
}

Array operator *(Array left, Array right){
    if(left.width != right.height){
        printf("Error : Unable to compute matrix multiple.\n");  
        fprintf(fpDebug, "Error : Unable to compute matrix multiple.\n");
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width); 
        exit(0);
    }
    Array res(left.height,right.width);
    for(int i = 0;i < res.height;i++){
        for(int j = 0;j < res.width;j++){
            for(int k = 0;k < left.width;k++){
                res.arr[i][j] += left.arr[i][k] * right.arr[k][j];
            }
        }
    }
    return res;
}

Array operator *(double left, Array right){
    Array res(right.height,right.width);
    for(int i = 0;i < res.height;i++){
        for(int j = 0;j < res.width;j++){
            res.arr[i][j] = right.arr[i][j] * left;
        }
    }
    return res;
}
