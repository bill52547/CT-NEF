# !/bin/zsh
!/bin/bash
# CUDA_ROOT=/usr/local/cuda-10.1
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# echo $CUDA_ROOT
# echo $TF_INC
# echo $TF_LIB

# $CUDA_ROOT/bin/nvcc add_one.cu -o add_one.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

cp /home/bill52547/Workspace/NefCT/nefct/cc/temp/* .

# g++ -std=c++11 add_one.cc add_one.cu.o -o add_one.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

bazel build -c opt //tensorflow/core/user_ops:tf_add_one.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"   --verbose_failures
# bazel build -c opt //tensorflow/core/user_ops:test_add_one.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --verbose_failures