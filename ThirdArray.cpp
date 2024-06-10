#include "DeepLearning.h"

ThirdArray::ThirdArray(void){
    depth = 0;
    width = 0;
    height = 0;
}

ThirdArray::ThirdArray(int dep, int hei, int wid){
    depth = dep;
    width = wid;
    height = hei;
    for(int i = 0;i < dep;i++){
        Array tmp(hei,wid);
        thiArr.push_back(tmp);
    }
}

void ThirdArray::ChangeSize(int d,int h,int w){
    depth = d;
    width = w;
    height = h;
    for(int i = 0;i < d;i++){
        Array tmp(h,w);
        thiArr.push_back(tmp);
    }
    return;
}

void ThirdArray::Bound(double mini,double maxi){
    for(int i = 0;i < depth;i++){
        thiArr[i].Bound(mini,maxi);
    }
    return;
}

bool ThirdArray::CheckFinite(void){
    bool check = true;
    for(int i = 0;i < depth;i++){
        if(!thiArr[i].CheckFinite()){
            fprintf(fpDebug, "Error : In Layer %d.\n",i);
            check = false;
            break;
        }
    }
    return check;
}

Array ThirdConv(ThirdArray inp,ThirdArray ker,int pad,int stride){
    int len_hei = (inp.height + 2 * pad - ker.height) / stride + 1;
    int len_wid = (inp.width + 2 * pad - ker.width) / stride + 1;
    Array res(len_hei,len_wid);
    for(int i = 0;i < inp.depth;i++){
        res = res + Conv(inp.thiArr[i], ker.thiArr[i], pad, stride);
    }
    return res;
}

void ThirdArray::PrintArray(FILE* feDebug){
    fprintf(fpDebug,"depth = %d\n",depth);
    for(int i = 0;i < depth;i++){
        thiArr[i].PrintArray(fpDebug);
    }
    return;
}

void ThirdArray::PrintMatlabArray(FILE* valid, string prefix){
    fprintf(valid,"%s = ",prefix.c_str());
    fprintf(valid,"[ ");
    for(int i = 0;i < depth;i++){
        thiArr[i].PrintMatlabArray(valid);
    }
    fprintf(valid,"];\n");
    return;
}

void ThirdArray::PrintArrayAvg(FILE* feDebug){
    fprintf(fpDebug,"depth = %d, height = %d, width = %d\n",depth,height,width);
    double avg = 0;
    double sqr = 0;
    double maxinum = -1e10;
    double mininum = 1e10;
    for(int n = 0;n < depth;n++){
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                if(thiArr[n].arr[i][j] > maxinum){
                    maxinum = thiArr[n].arr[i][j];
                }
                if(thiArr[n].arr[i][j] < mininum){
                    mininum = thiArr[n].arr[i][j];
                }
                avg += thiArr[n].arr[i][j];
            }
        }
    }
    avg /= (depth * height * width);
    for(int n = 0;n < depth;n++){
        for(int i = 0;i < height;i++){
            for(int j = 0;j < width;j++){
                sqr += (thiArr[n].arr[i][j] - avg) * (thiArr[n].arr[i][j] - avg);
            }
        }
    }
    sqr /= (depth * height * width);
    fprintf(fpDebug,"avg = %llf, sqr = %llf, maxinum = %llf, mininum = %llf\n",avg,sqr,maxinum,mininum);
    return;
}

ThirdArray sqrt(ThirdArray inp){
    ThirdArray res(inp.depth,inp.height,inp.width);
    for(int i = 0;i < inp.depth;i++){
        res.thiArr[i] = sqrt(inp.thiArr[i]);
    }
    return res;
}

double ThirdArray::addTogether(void){
    double res = 0;
    for(int i = 0;i < depth;i++){
        res += thiArr[i].addTogether();
    }
    return res;
}

ThirdArray DotProduct(ThirdArray left,ThirdArray right){
    if(left.depth != right.depth || left.width != right.width || left.height != right.height){
        printf("Error: Unable to compute DotProduct\n");
        exit(0);
    }
    ThirdArray res(left.depth, left.height, left.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = DotProduct(left.thiArr[i], right.thiArr[i]);
    }
    return res;
}

void ThirdArray::operator =(ThirdArray right){
    if(depth == 0 && width == 0 && height == 0){
        ChangeSize(right.depth, right.height, right.width);
    }
    else if(depth != right.depth || width != right.width || height != right.height){
        printf("Error : Unable to equal third matrix\n");
        fprintf(fpDebug, "Error : Unable to equal third matrix\n");
        fprintf(fpDebug, "depth = %d\n",depth);
        fprintf(fpDebug, "right.depth = %d\n",right.depth);
        fprintf(fpDebug, "height = %d\n",height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "width = %d\n",width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    for(int i = 0;i < depth;i++){
        thiArr[i] = right.thiArr[i];
    }
    return;
}

void ThirdArray::operator =(double right){
    for(int i = 0;i < depth;i++){
        thiArr[i] = right;
    }
    return;
}


ThirdArray operator +(ThirdArray left, double num){
    ThirdArray res(left.depth, left.height, left.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = left.thiArr[i] + num;
    }
    return res;
}

ThirdArray operator +(ThirdArray left, ThirdArray right){
    if(left.depth != right.depth || left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "left.depth = %d\n",left.depth);
        fprintf(fpDebug, "right.depth = %d\n",right.depth);
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    ThirdArray res(left.depth, left.height, left.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = left.thiArr[i] + right.thiArr[i];
    }
    return res;
}

ThirdArray operator -(ThirdArray left, ThirdArray right){
    if(left.depth != right.depth || left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "left.depth = %d\n",left.depth);
        fprintf(fpDebug, "right.depth = %d\n",right.depth);
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    ThirdArray res(left.depth, left.height, left.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = left.thiArr[i] - right.thiArr[i];
    }
    return res;
}

ThirdArray operator *(double left, ThirdArray right){
    ThirdArray res(right.depth, right.height, right.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = left * right.thiArr[i];
    }
    return res;
}

ThirdArray operator /(ThirdArray left, ThirdArray right){
    if(left.depth != right.depth || left.width != right.width || left.height != right.height){
        printf("Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "Error : Unable to compute third matrix adding\n");
        fprintf(fpDebug, "left.depth = %d\n",left.depth);
        fprintf(fpDebug, "right.depth = %d\n",right.depth);
        fprintf(fpDebug, "left.height = %d\n",left.height);
        fprintf(fpDebug, "right.height = %d\n",right.height);
        fprintf(fpDebug, "left.width = %d\n",left.width);
        fprintf(fpDebug, "right.width = %d\n",right.width);
        exit(0);
    }
    ThirdArray res(left.depth, left.height, left.width);
    for(int i = 0;i < res.depth;i++){
        res.thiArr[i] = left.thiArr[i] / right.thiArr[i];
    }
    return res;
}