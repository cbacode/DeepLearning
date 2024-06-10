# DeepLearning仓库使用说明

## 一. 模型的使用

### 1. mingw环境

模型需要mingw环境运行，请先确认您已经下载了mingw，并且将mingw所在的文件夹添加到了环境变量中。

### 2. 确认环境

使用gcc -v命令检查gcc能否正确运行。如果可以正确运行，运行结果应输出类似以下文字

Using built-in specs.
COLLECT_GCC=C:\msys64\mingw64\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/msys64/mingw64/bin/../lib/gcc/x86_64-w64-mingw32/12.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../gcc-12.2.0/configure --prefix=/mingw64 --with-local-prefix=/mingw64/local --build=x86_64-w64-mingw32 --host=x86_64-w64-mingw32 --target=x86_64-w64-mingw32 --with-native-system-header-dir=/mingw64/include --libexecdir=/mingw64/lib --enable-bootstrap --enable-checking=release --with-arch=x86-64 --with-tune=generic --enable-languages=c,lto,c++,fortran,ada,objc,obj-c++,jit --enable-shared --enable-static --enable-libatomic --enable-threads=posix --enable-graphite --enable-fully-dynamic-string --enable-libstdcxx-filesystem-ts --enable-libstdcxx-time --disable-libstdcxx-pch --enable-lto --enable-libgomp --disable-multilib --disable-rpath --disable-win32-registry --disable-nls --disable-werror --disable-symvers --with-libiconv --with-system-zlib --with-gmp=/mingw64 --with-mpfr=/mingw64 --with-mpc=/mingw64 --with-isl=/mingw64 --with-pkgversion='Rev6, Built by MSYS2 project' --with-bugurl=https://github.com/msys2/MINGW-packages/issues --with-gnu-as --with-gnu-ld --disable-libstdcxx-debug --with-boot-ldflags=-static-libstdc++ --with-stage1-ldflags=-static-libstdc++
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 12.2.0 (Rev6, Built by MSYS2 project)

注意此处Thread model: posix是否一致，若不一致请下载https://gitee.com/CreateMe/mingw-std-threads 中的文件并放置到mingw中头文件对应文件夹, 并打开DeepLearning.h并更改<thread.h>为<mingw32.thread.h>

注：此处参考https://blog.csdn.net/qq_43478653/article/details/115369025

使用mingw32-make --v检查mingw32-make能否正常运行，若能，应得到类似如下输出。

GNU Make 4.4
Built for Windows32
Copyright (C) 1988-2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

### 3. 选择运行模型

在mainv1.cpp - mainv5.cpp(mainv4.cpp除外)选择一个main函数复制粘贴到main.cpp

注：运行SR模式(mainv5.cpp)需要进行数据准备

在sr文件夹的train文件夹下有对应url，下载后解压到该文件夹下，之后运行“dataprovider.py”生成对应的bin后缀文件

### 4. 使用mingw32运行代码

在代码所在文件夹下运行mingw32-make，再运行./main即可

### 5. 运行模型中的pytorch模型

完成第三步的数据准备工作后，在pytorch文件夹打开对应的“readme.md”，根据该文件的提示训练，在model文件夹中预留了一个预训练模型

## 二. 模型的缺陷

1. 目前模型的最终层只能为SoftMax层，否则反向传播可能失效，同时模型的BatchNormalization Layer尚未通过梯度检验，因此请不要使用mainv4.cpp中的模型

2. 目前模型没有图形化输出结果的能力

3. 模型中层与层之间需要自行添加linear层转换输入输出个数，后续可能会进行合并

## 三. 其他问题

1. 若依然无法运行模型或者对于模型的运行速度优化有想法的同学可以联系作者

2. 目前该模型的运行时间大致为matlab的25倍，因此运行前请注意时间消耗
