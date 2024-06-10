## 一. 模型的使用

# 1. mingw环境

模型需要mingw环境运行，请先确认您已经下载了mingw，并且将mingw所在的文件夹添加到了环境变量中。

# 2. 确认环境

使用gcc -v命令检查gcc能否正确运行。如果可以正确运行，运行结果应输出类似以下文字

Using built-in specs.
COLLECT_GCC=C:\msys64\mingw64\bin\gcc.exe
COLLECT_LTO_WRAPPER=C:/msys64/mingw64/bin/../lib/gcc/x86_64-w64-mingw32/12.2.0/lto-wrapper.exe
Target: x86_64-w64-mingw32
Configured with: ../gcc-12.2.0/configure --prefix=/mingw64 --with-local-prefix=/mingw64/local --build=x86_64-w64-mingw32 --host=x86_64-w64-mingw32 --target=x86_64-w64-mingw32 --with-native-system-header-dir=/mingw64/include --libexecdir=/mingw64/lib --enable-bootstrap --enable-checking=release --with-arch=x86-64 --with-tune=generic --enable-languages=c,lto,c++,fortran,ada,objc,obj-c++,jit --enable-shared --enable-static --enable-libatomic --enable-threads=posix --enable-graphite --enable-fully-dynamic-string --enable-libstdcxx-filesystem-ts --enable-libstdcxx-time --disable-libstdcxx-pch --enable-lto --enable-libgomp --disable-multilib --disable-rpath --disable-win32-registry --disable-nls --disable-werror --disable-symvers --with-libiconv --with-system-zlib --with-gmp=/mingw64 --with-mpfr=/mingw64 --with-mpc=/mingw64 --with-isl=/mingw64 --with-pkgversion='Rev6, Built by MSYS2 project' --with-bugurl=https://github.com/msys2/MINGW-packages/issues --with-gnu-as --with-gnu-ld --disable-libstdcxx-debug --with-boot-ldflags=-static-libstdc++ --with-stage1-ldflags=-static-libstdc++
Thread model: posix
Supported LTO compression algorithms: zlib zstd
gcc version 12.2.0 (Rev6, Built by MSYS2 project)

使用mingw32-make --v检查mingw32-make能否正常运行，若能，应得到类似如下输出。

GNU Make 4.4
Built for Windows32
Copyright (C) 1988-2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

# 3. 使用mingw32运行代码

在代码所在文件夹下运行mingw32-make，再运行./main即可

## 二. 模型的缺陷

# 1. 目前模型的最终层只能为SoftMax层，否则反向传播可能失效

# 2. 目前模型没有图形化输出结果的能力

# 3. 模型中层与层之间需要自行添加linear层转换输入输出个数，后续可能会进行合并