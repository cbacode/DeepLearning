#include "DeepLearning.h"

int GetNumber(ifstream* fp){
    unsigned char bytes[2];
    unsigned int sum = 0;
    fp->read((char*)bytes, 2);
    sum += bytes[1] | (bytes[0]<<8);
    sum = sum << 16;
    fp->read((char*)bytes, 2);
    sum += bytes[1] | (bytes[0]<<8);
    return sum;
}

void GetPicture(ifstream* fp){
    unsigned char tmp;
    for(int i = 0;i < maxHeight;i++){
        for(int j = 0;j < maxWidth;j++){
            pic[i][j] = 0;
        }
    }
    int num = 0;
    if(inputHeight > maxHeight || inputWidth > maxWidth){
        printf("Error : Unable to store whole picture\n");
        exit(0);
    }
    for(int i = 0;i < inputHeight;i++){
        for(int j = 0;j < inputWidth;j++){
            fp->read((char*)&tmp,sizeof(tmp));
            num = (int)tmp;
            pic[i][j] = num;
            //fprintf(fpDebug, "%d ", num);
        }
        //fprintf(fpDebug, ";\n");
    }

    bool check = false;
    for(int i = 0;i < inputHeight;i++){
        for(int j = 0;j < inputWidth;j++){
            if(pic[i][j] != 0){
                check = true;
            }
        }
    }
    if(!check){
        printf("Error : Reading empty picture!\n");
        exit(0);
    }
    return;
}

void GetLabel(ifstream* fp, int len){
    if(len > maxNumLabel){
        printf("len = %d\n",len);
        printf("Error : Unable to get whole label\n");
        exit(0);
    }
    unsigned char tmp = '\0';
    int num = 0;
    for(int i = 0;i < len;i++){
        fp->read((char*)&tmp, sizeof(tmp));
        num = (int)tmp;
        label[i] = num;
        //fprintf(fpDebug, "%d ", num);
    }
    //fprintf(fpDebug, "\n");
    //fflush(fpDebug);
    return;
}

void DecreaseLearningRate(int epoch){
    if(epoch % LearningChange == 0 && epoch != 0){
        LearningRate = LearningRate * LearningRatio;
        printf("Learning rate : %llf\n",LearningRate);
        fprintf(fpResult, "Learning rate : %llf\n",LearningRate);
    }
}

double DotProduct(double left, double right){
    return left * right;
}

void TestArray(FILE* fpDebug){
    Array A(4,3);
    Array B(3,4);
    Array C(4,4);
    Array D(3,3);

    // To check the size of new array
    // printf("C.arr.size() = %d\n",C.arr.size());

    for(int i = 0;i < A.arr.size();i++){
        for(int j = 0;j < A.arr[i].size();j++){
            A.arr[i][j] = i;
        }
    }

    for(int i = 0;i < B.arr.size();i++){
        for(int j = 0;j < B.arr[i].size();j++){
            B.arr[i][j] = j;
        }
    }

    // To check the result of dot product
    // C = DotProduct(A,B);

    // To ensure left multiple and right multiple is different
    // and arranged fine
    C = A * B;
    D = B * A;
    fprintf(fpDebug,"A.arr = \n");
    A.PrintArray(fpDebug);
    fprintf(fpDebug,"B.arr = \n");
    B.PrintArray(fpDebug);
    fprintf(fpDebug,"C.arr = \n");
    C.PrintArray(fpDebug);
    fprintf(fpDebug,"D.arr = \n");
    D.PrintArray(fpDebug);
    fflush(fpDebug);
    exit(0);
}

void TestSign(FILE* fpDebug){
    double test = 1;
    int sign = (test > 0) ? 1 : -1;
    fprintf(fpDebug,"%llf %d\n",test, sign);
    exit(0);
}

void TestSingleConv(FILE* fpDebug){
    double inp[7][7] = {
        {2,3,7,4,6,2,9},
        {6,6,9,8,7,4,3},
        {3,4,8,3,8,9,7},
        {7,8,3,6,6,3,4},
        {4,2,1,8,3,4,6},
        {3,2,4,1,9,8,3},
        {0,1,3,9,2,1,4} 
    };
    Array input(7,7);
    for(int i = 0;i < 7;i++){
        for(int j = 0;j < 7;j++){
            input.arr[i][j] = inp[i][j];
        }
    }

    double ker[3][3] = {
        {3,4,4},
        {1,0,2},
        {-1,0,3}
    };
    Array kernal(3,3);
    for(int i = 0;i < 3;i++){
        for(int j = 0;j < 3;j++){
            kernal.arr[i][j] = ker[i][j];
        }
    }

    Array res = Conv(input,kernal,2,2);
    res.PrintArray(fpDebug);
    exit(0);
}

void TestSpread(FILE* fpDebug){
    double inp[7][7] = {
        {2,3,7,4,6,2,9},
        {6,6,9,8,7,4,3},
        {3,4,8,3,8,9,7},
        {7,8,3,6,6,3,4},
        {4,2,1,8,3,4,6},
        {3,2,4,1,9,8,3},
        {0,1,3,9,2,1,4} 
    };
    Array input(7,7);
    for(int i = 0;i < 7;i++){
        for(int j = 0;j < 7;j++){
            input.arr[i][j] = inp[i][j];
        }
    }
    Array output = input.Spread(1,7,7);
    input.PrintArray(fpDebug);
    output.PrintArray(fpDebug);
    exit(0);
}

void TestDiffConv(FILE* fpDebug){
    int numKernel = 4;
    int padding = 1;
    int stride = 1;
    int kerDepth = 4;
    int kerHeight = 3;
    int kerWidth = 3;
    int inpDepth = 4;
    int inpHeight = 10;
    int inpWidth = 10;
    double step = 1e-7;
    double bound = 1e-5;

    vector<ThirdArray> inp;
    vector<ThirdArray> inpLeft;
    vector<ThirdArray> inpRight;
    vector<ThirdArray> out;
    vector<ThirdArray> outLeft;
    vector<ThirdArray> outRight;
    vector<ThirdArray> err;

    // use loss function as sum(B.*dB)
    // dB will be randomly produced
    // so loss function is different
    vector<double> lossLeft;
    vector<double> lossRight;

    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    ThirdArray tmp(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < inpDepth;j++){
            for(int k = 0;k < inpHeight;k++){
                for(int l = 0;l < inpWidth;l++){
                    tmp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        inp.push_back(tmp);
        inpLeft.push_back(tmp);
        inpRight.push_back(tmp);
    }
    lossLeft.resize(batchSize);
    lossRight.resize(batchSize);

    ConvLayer conv(inpDepth,inpHeight,inpWidth, kerDepth,kerHeight,kerWidth, numKernel,padding,stride);

    int outDepth = conv.outDepth;
    int outHeight = conv.outHeight;
    int outWidth = conv.outWidth;
    ThirdArray temp(outDepth,outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    temp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        err.push_back(temp);
        outLeft.push_back(temp);
        outRight.push_back(temp);
    }

    fprintf(fpDebug,"Testing InpDiff\n");
    printf("Testing InpDiff\n");
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                inpLeft = inp;
                inpRight = inp;
                for(int i = 0;i < batchSize;i++){
                    inpLeft[i].thiArr[j].arr[k][l] -= step;
                    inpRight[i].thiArr[j].arr[k][l] += step;
                }
                conv.forward(inpLeft);
                outLeft = conv.output;
                conv.forward(inpRight);
                outRight = conv.output;
                for(int i = 0;i < batchSize;i++){
                    // ThirdArray temp(outDepth,outHeight,outWidth);
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                }
                
                conv.forward(inp);
                conv.backward(err);
                bool check = true;
                int cnt = 0;
                for(int i = 0;i < batchSize;i++){
                    double backward = conv.error[i].thiArr[j].arr[k][l] * 2;
                    double forward = (lossRight[i] - lossLeft[i]) / step;
                    double res = (backward - forward) / (backward + forward);
                    if(abs(res) > bound){
                        fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                        check = false;
                        cnt = cnt + 1;
                    }
                }
                if(check){
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }

    fflush(fpDebug);
    fprintf(fpDebug,"Testing KerDiff\n");
    printf("Testing KerDiff\n");
    for(int i = 0;i < numKernel;i++){
        for(int j = 0;j < kerDepth;j++){
            for(int k = 0;k < kerHeight;k++){
                for(int l = 0;l < kerWidth;l++){
                    fprintf(fpDebug,"Test for i = %d, j = %d, k = %d, l = %d\n",i,j,k,l);
                    double ori = conv.kernal[i].thiArr[j].arr[k][l];
                    conv.kernal[i].thiArr[j].arr[k][l] = ori - step;
                    conv.forward(inp);
                    outLeft = conv.output;
                    conv.kernal[i].thiArr[j].arr[k][l] = ori + step;
                    conv.forward(inp);
                    outRight = conv.output;
                    conv.kernal[i].thiArr[j].arr[k][l] = ori;
                    for(int m = 0;m < batchSize;m++){
                        // ThirdArray temp(outDepth,outHeight,outWidth);
                        temp = DotProduct(outLeft[m],err[m]);
                        lossLeft[m] = temp.addTogether();
                        temp = DotProduct(outRight[m],err[m]);
                        lossRight[m] = temp.addTogether();
                    }
                    conv.forward(inp);
                    conv.backward(err);
                    bool check = true;
                    int cnt = 0;
                    double backward = conv.diffKer[i].thiArr[j].arr[k][l] * 2;
                    double forward = 0;
                    for(int m = 0;m < batchSize;m++){
                        forward += (lossRight[m] - lossLeft[m]);    
                    }
                    forward /= batchSize;
                    forward /= step;
                    double res = (backward - forward) / (backward + forward);
                    if(abs(res) > bound){
                        fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                        check = false;
                        cnt = cnt + 1;
                    }
                    if(check){
                        fprintf(fpDebug,"OK\n");
                    }
                    else{
                        fprintf(fpDebug,"err points : %d\n",cnt);
                    }
                }
            }
        }
    }

    fflush(fpDebug);
    fprintf(fpDebug,"Testing BiasDiff\n");
    printf("Testing BiasDiff\n");
    for(int i = 0;i < numKernel;i++){
        fprintf(fpDebug,"Test for i = %d\n",i);
        double ori = conv.bias[i];
        conv.bias[i] = ori - step;
        conv.forward(inp);
        outLeft = conv.output;
        conv.bias[i] = ori + step;
        conv.forward(inp);
        outRight = conv.output;
        conv.bias[i] = ori;
        for(int m = 0;m < batchSize;m++){
            // ThirdArray temp(outDepth,outHeight,outWidth);
            temp = DotProduct(outLeft[m],err[m]);
            lossLeft[m] = temp.addTogether();
            temp = DotProduct(outRight[m],err[m]);
            lossRight[m] = temp.addTogether();
        }
        conv.forward(inp);
        conv.backward(err);
        bool check = true;
        int cnt = 0;
        double backward = conv.diffBias[i] * 2;
        double forward = 0;
        for(int m = 0;m < batchSize;m++){
            forward += (lossRight[m] - lossLeft[m]);    
        }
        forward /= batchSize;
        forward /= step;
        double res = (backward - forward) / (backward + forward);
        if(abs(res) > bound){
            fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
            check = false;
            cnt = cnt + 1;
        }
        if(check){
            fprintf(fpDebug,"OK\n");
        }
        else{
            fprintf(fpDebug,"err points : %d\n",cnt);
        }
    }
    exit(0);
}

void TestDiffLinear(FILE* fpDebug){
    int inpHeight = 20;
    int inpWidth = 1;
    int outHeight = 10;
    int outWidth = 1;
    double step = 1e-7;
    double bound = 1e-5;

    vector<Array> inp;
    vector<Array> inpLeft;
    vector<Array> inpRight;
    vector<Array> out;
    vector<Array> outLeft;
    vector<Array> outRight;
    vector<Array> err;

    vector<double> lossLeft;
    vector<double> lossRight;

    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    Array tmp(inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                tmp.arr[k][l] = nor(e);
            }
        }
        inp.push_back(tmp);
        inpLeft.push_back(tmp);
        inpRight.push_back(tmp);
    }
    lossLeft.resize(batchSize);
    lossRight.resize(batchSize);

    LinearLayer line(inpHeight,outHeight);

    Array temp(outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        err.push_back(temp);
        out.push_back(temp);
        outLeft.push_back(temp);
        outRight.push_back(temp);
    }

    printf("Testing Input diff\n");
    fprintf(fpDebug, "Testing Input diff\n");
    for(int k = 0;k < inpHeight;k++){
        for(int l = 0;l < inpWidth;l++){
            //printf("Test for k = %d, l = %d\n",k,l);
            fprintf(fpDebug, "Test for k = %d, l = %d\n",k,l);
            inpLeft = inp;
            inpRight = inp;
            for(int i = 0;i < batchSize;i++){
                inpLeft[i].arr[k][l] -= step;
                inpRight[i].arr[k][l] += step;
            }
            line.forward(inpLeft);
            outLeft = line.output;
            line.forward(inpRight);
            outRight = line.output;
            line.backward(err);
            for(int i = 0;i < batchSize;i++){
                // Array temp(outHeight,outWidth);
                temp = DotProduct(outLeft[i],err[i]);
                lossLeft[i] = temp.addTogether();
                temp = DotProduct(outRight[i],err[i]);
                lossRight[i] = temp.addTogether();
            }
            bool check = true;
            int cnt = 0;
            for(int i = 0;i < batchSize;i++){
                double backward = line.error[i].arr[k][l] * 2;
                double forward = (lossRight[i] - lossLeft[i]) / step;
                double res = (backward - forward) / (backward + forward);
                if(abs(res) > eps){
                    fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                    fflush(fpDebug);
                    check = false;
                    cnt = cnt + 1;
                }
            }
            if(check){
                fprintf(fpDebug,"OK\n");
            }
            else{
                fprintf(fpDebug,"err points : %d\n",cnt);
            }
        }
    }

    printf("Testing Coff diff\n");
    fprintf(fpDebug, "Testing Coff diff\n");
    for(int i = 0;i < outHeight;i++){
        for(int j = 0;j < inpHeight;j++){
            fprintf(fpDebug,"Test for i = %d, j = %d\n", i, j);
            double ori = line.coff.arr[i][j];
            line.coff.arr[i][j] = ori - step;
            line.forward(inp);
            outLeft = line.output;
            line.coff.arr[i][j] = ori + step;
            line.forward(inp);
            outRight = line.output;
            line.coff.arr[i][j] = ori;
            for(int m = 0;m < batchSize;m++){
                // ThirdArray temp(outDepth,outHeight,outWidth);
                temp = DotProduct(outLeft[m],err[m]);
                lossLeft[m] = temp.addTogether();
                temp = DotProduct(outRight[m],err[m]);
                lossRight[m] = temp.addTogether();
            }
            line.forward(inp);
            line.backward(err);
            bool check = true;
            int cnt = 0;
            double backward = line.diffCoff.arr[i][j] * 2;
            double forward = 0;
            for(int i = 0;i < batchSize;i++){
                forward += (lossRight[i] - lossLeft[i]);    
            }
            forward /= batchSize;
            forward /= step;
            double res = (backward - forward) / (backward + forward);
            if(abs(res) > bound){
                fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                check = false;
                cnt = cnt + 1;
            }
            if(check){
                fprintf(fpDebug,"OK\n");
            }
            else{
                fprintf(fpDebug,"err points : %d\n",cnt);
            }
        }    
    }
    
    printf("Testing Bias diff\n");
    fprintf(fpDebug, "Testing Bias diff\n");
    for(int i = 0;i < outHeight;i++){
        fprintf(fpDebug,"Test for i = %d\n",i);
        double ori = line.bias.arr[i][0];
        line.bias.arr[i][0] = ori - step;
        line.forward(inp);
        outLeft = line.output;
        line.bias.arr[i][0] = ori + step;
        line.forward(inp);
        outRight = line.output;
        line.bias.arr[i][0] = ori;
        for(int m = 0;m < batchSize;m++){
            // ThirdArray temp(outDepth,outHeight,outWidth);
            temp = DotProduct(outLeft[m],err[m]);
            lossLeft[m] = temp.addTogether();
            temp = DotProduct(outRight[m],err[m]);
            lossRight[m] = temp.addTogether();
        }
        line.forward(inp);
        line.backward(err);
        bool check = true;
        int cnt = 0;
        double backward = line.diffBias.arr[i][0] * 2;
        double forward = 0;
        for(int m = 0;m < batchSize;m++){
            forward += (lossRight[m] - lossLeft[m]);    
        }
        forward /= batchSize;
        forward /= step;
        double res = (backward - forward) / (backward + forward);
        if(abs(res) > bound){
            fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
            check = false;
            cnt = cnt + 1;
        }
        if(check){
            fprintf(fpDebug,"OK\n");
        }
        else{
            fprintf(fpDebug,"err points : %d\n",cnt);
        }
    }
    exit(0);
}

void TestDiffBatchNorm(FILE* fpDebug){
    int inpDepth = 2;
    int inpHeight = 4;
    int inpWidth = 3;
    double step = 1e-7;
    double bound = 1e-5;

    vector<ThirdArray> inp;
    vector<ThirdArray> inpLeft;
    vector<ThirdArray> inpRight;
    vector<ThirdArray> out;
    vector<ThirdArray> outLeft;
    vector<ThirdArray> outRight;
    vector<ThirdArray> err;

    // use loss function as sum(B.*dB)
    // dB will be randomly produced
    // so loss function is different
    vector<double> lossLeft;
    vector<double> lossRight;

    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    ThirdArray tmp(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < inpDepth;j++){
            for(int k = 0;k < inpHeight;k++){
                for(int l = 0;l < inpWidth;l++){
                    tmp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        inp.push_back(tmp);
        inpLeft.push_back(tmp);
        inpRight.push_back(tmp); 
    }
    lossLeft.resize(batchSize);
    lossRight.resize(batchSize);

    BatchLayer batch(inpDepth,inpHeight,inpWidth);

    int outDepth = inpDepth;
    int outHeight = inpHeight;
    int outWidth = inpWidth;
    ThirdArray temp(outDepth,outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    temp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        err.push_back(temp);
        outLeft.push_back(temp);
        outRight.push_back(temp);
    }

    // Check Input diff
    printf("Testing Input diff\n");
    fprintf(fpDebug, "Testing Input diff\n");
    inpLeft = inp;
    inpRight = inp;
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                bool check = true;
                int cnt = 0;
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                for(int i = 0;i < batchSize;i++){
                    inpLeft[i] = inp[i];
                    inpRight[i] = inp[i];
                    inpLeft[i].thiArr[j].arr[k][l] -= step;
                    inpRight[i].thiArr[j].arr[k][l] += step;
                
                    batch.forward(inpLeft);
                    outLeft = batch.output;
                    batch.forward(inpRight);
                    outRight = batch.output;
                    // ThirdArray temp(outDepth,outHeight,outWidth);
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                
                    batch.forward(inp);
                    batch.backward(err);
                    
                    double backward = batch.error[i].thiArr[j].arr[k][l] * 2;
                    double forward = (lossRight[i] - lossLeft[i]);
                    forward /= step;
                    forward /= batchSize;
                    double res = (backward - forward) / (backward + forward);
                    if(abs(res) > bound){
                        fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                        check = false;
                        cnt = cnt + 1;
                    }
                }
                if(check){
                    //printf("OK\n");
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    //printf("err points : %d\n",cnt);
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }

    printf("Testing Gamma diff\n");
    fprintf(fpDebug, "Testing Gamma diff\n");
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                bool check = true;
                int cnt = 0;
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                double ori = batch.gamma.thiArr[j].arr[k][l];
                batch.gamma.thiArr[j].arr[k][l] = ori - step;
                batch.testForward(inp);
                outLeft = batch.output;
                batch.gamma.thiArr[j].arr[k][l] = ori + step;
                batch.testForward(inp);
                outRight = batch.output;
                batch.gamma.thiArr[j].arr[k][l] = ori;
                
                for(int i = 0;i < batchSize;i++){
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                }

                batch.forward(inp);
                batch.backward(err);
                    
                double backward = batch.diffGamma.thiArr[j].arr[k][l] * 2;
                double forward = 0;
                for(int i = 0;i < batchSize;i++){
                    forward += (lossRight[i] - lossLeft[i]);
                }
                forward /= step;
                forward /= batchSize;
                double res = (backward - forward) / (backward + forward);
                if(abs(res) > bound){
                    fprintf(fpDebug,"backward = %llf, forward = %llf, res = %llf\n",backward,forward,res);
                    check = false;
                    cnt = cnt + 1;
                }
                
                if(check){
                    //printf("OK\n");
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    //printf("err points : %d\n",cnt);
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }

    printf("Testing Beta diff\n");
    fprintf(fpDebug, "Testing Beta diff\n");
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                bool check = true;
                int cnt = 0;
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                double ori = batch.beta.thiArr[j].arr[k][l];
                batch.beta.thiArr[j].arr[k][l] = ori - step;
                batch.testForward(inp);
                outLeft = batch.output;
                batch.beta.thiArr[j].arr[k][l] = ori + step;
                batch.testForward(inp);
                outRight = batch.output;
                batch.beta.thiArr[j].arr[k][l] = ori;

                batch.forward(inp);
                batch.backward(err);
                
                for(int i = 0;i < batchSize;i++){
                    // ThirdArray temp(outDepth,outHeight,outWidth);
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                }

                batch.forward(inp);
                batch.backward(err);
                    
                double backward = batch.diffBeta.thiArr[j].arr[k][l] * 2;
                double forward = 0;
                for(int i = 0;i < batchSize;i++){
                    forward += (lossRight[i] - lossLeft[i]);
                }
                forward /= step;
                forward /= batchSize;
                double res = (backward - forward) / (backward + forward);
                if(abs(res) > bound){
                    fprintf(fpDebug,"backward = %llf, forward = %llf, res = %llf\n",backward,forward,res);
                    check = false;
                    cnt = cnt + 1;
                }
                
                if(check){
                    //printf("OK\n");
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    //printf("err points : %d\n",cnt);
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }
    exit(0);
}

void TestMaxPooling(FILE* fpDebug){
    int inpDepth = 2;
    int inpHeight = 16;
    int inpWidth = 16;

    int kerDepth = inpDepth;
    int kerHeight = 2;
    int kerWidth = 2;
    int stride = 2;
    double step = 1e-6;
    double bound = 1e-5;

    vector<ThirdArray> inp;
    vector<ThirdArray> inpLeft;
    vector<ThirdArray> inpRight;
    vector<ThirdArray> out;
    vector<ThirdArray> outLeft;
    vector<ThirdArray> outRight;
    vector<ThirdArray> err;

    // use loss function as sum(B.*dB)
    // dB will be randomly produced
    // so loss function is different
    vector<double> lossLeft;
    vector<double> lossRight;

    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    ThirdArray tmp(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < inpDepth;j++){
            for(int k = 0;k < inpHeight;k++){
                for(int l = 0;l < inpWidth;l++){
                    tmp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        inp.push_back(tmp);
        inpLeft.push_back(tmp);
        inpRight.push_back(tmp);
    }
    lossLeft.resize(batchSize);
    lossRight.resize(batchSize);

    BatchLayer batch(inpDepth,inpHeight,inpWidth);

    int outDepth = inpDepth;
    int outHeight = inpHeight / stride;
    int outWidth = inpWidth / stride;
    ThirdArray temp(outDepth,outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    temp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        err.push_back(temp);
        outLeft.push_back(temp);
        outRight.push_back(temp);
    }

    MaxPoolingLayer pool(inpDepth,inpHeight,inpWidth,kerDepth,kerHeight,kerWidth,stride,1);
    fprintf(fpDebug,"Testing InpDiff\n");
    printf("Testing InpDiff\n");
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                inpLeft = inp;
                inpRight = inp;
                for(int i = 0;i < batchSize;i++){
                    inpLeft[i].thiArr[j].arr[k][l] -= step;
                    inpRight[i].thiArr[j].arr[k][l] += step;
                }
                pool.forward(inpLeft);
                outLeft = pool.output;
                pool.forward(inpRight);
                outRight = pool.output;
                for(int i = 0;i < batchSize;i++){
                    // ThirdArray temp(outDepth,outHeight,outWidth);
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                }
                
                pool.forward(inp);
                pool.backward(err);
                bool check = true;
                int cnt = 0;
                for(int i = 0;i < batchSize;i++){
                    double backward = pool.error[i].thiArr[j].arr[k][l] * 2;
                    double forward = (lossRight[i] - lossLeft[i]) / step;
                    double res = (backward - forward) / (backward + forward);
                    if(abs(res) > bound){
                        fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                        check = false;
                        cnt = cnt + 1;
                    }
                }
                if(check){
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }
    exit(0);
}

void TestThiRel(FILE* fpDebug){
    int inpDepth = 2;
    int inpHeight = 16;
    int inpWidth = 16;

    int kerDepth = inpDepth;
    int kerHeight = 2;
    int kerWidth = 2;
    int stride = 2;
    double step = 1e-6;
    double bound = 1e-4;

    vector<ThirdArray> inp;
    vector<ThirdArray> inpLeft;
    vector<ThirdArray> inpRight;
    vector<ThirdArray> out;
    vector<ThirdArray> outLeft;
    vector<ThirdArray> outRight;
    vector<ThirdArray> err;

    // use loss function as sum(B.*dB)
    // dB will be randomly produced
    // so loss function is different
    vector<double> lossLeft;
    vector<double> lossRight;

    default_random_engine e; 
	normal_distribution<double> nor(0, 1);
    ThirdArray tmp(inpDepth,inpHeight,inpWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < inpDepth;j++){
            for(int k = 0;k < inpHeight;k++){
                for(int l = 0;l < inpWidth;l++){
                    tmp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        inp.push_back(tmp);
        inpLeft.push_back(tmp);
        inpRight.push_back(tmp);
    }
    lossLeft.resize(batchSize);
    lossRight.resize(batchSize);

    int outDepth = inpDepth;
    int outHeight = inpHeight;
    int outWidth = inpWidth;
    ThirdArray temp(outDepth,outHeight,outWidth);
    for(int i = 0;i < batchSize;i++){
        for(int j = 0;j < outDepth;j++){
            for(int k = 0;k < outHeight;k++){
                for(int l = 0;l < outWidth;l++){
                    temp.thiArr[j].arr[k][l] = nor(e);
                }
            }
        }
        err.push_back(temp);
        outLeft.push_back(temp);
        outRight.push_back(temp);
    }

    ThirdReLULayer thi(inpDepth,inpHeight,inpWidth,0);
    fprintf(fpDebug,"Testing InpDiff\n");
    printf("Testing InpDiff\n");
    for(int j = 0;j < inpDepth;j++){
        for(int k = 0;k < inpHeight;k++){
            for(int l = 0;l < inpWidth;l++){
                //printf("Test for j = %d, k = %d, l = %d\n",j,k,l);
                fprintf(fpDebug,"Test for j = %d, k = %d, l = %d\n",j,k,l);
                inpLeft = inp;
                inpRight = inp;
                for(int i = 0;i < batchSize;i++){
                    inpLeft[i].thiArr[j].arr[k][l] -= step;
                    inpRight[i].thiArr[j].arr[k][l] += step;
                }
                thi.forward(inpLeft);
                outLeft = thi.output;
                thi.forward(inpRight);
                outRight = thi.output;
                for(int i = 0;i < batchSize;i++){
                    // ThirdArray temp(outDepth,outHeight,outWidth);
                    temp = DotProduct(outLeft[i],err[i]);
                    lossLeft[i] = temp.addTogether();
                    temp = DotProduct(outRight[i],err[i]);
                    lossRight[i] = temp.addTogether();
                }
                
                thi.forward(inp);
                thi.backward(err);
                bool check = true;
                int cnt = 0;
                for(int i = 0;i < batchSize;i++){
                    double backward = thi.error[i].thiArr[j].arr[k][l] * 2;
                    double forward = (lossRight[i] - lossLeft[i]) / step;
                    double res = (backward - forward) / (backward + forward);
                    if(abs(res) > bound){
                        fprintf(fpDebug,"point %d: backward = %llf, forward = %llf, res = %llf\n",i,backward,forward,res);
                        check = false;
                        cnt = cnt + 1;
                    }
                }
                if(check){
                    fprintf(fpDebug,"OK\n");
                }
                else{
                    fprintf(fpDebug,"err points : %d\n",cnt);
                }
            }
        }
    }
    exit(0);
}