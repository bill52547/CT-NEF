#!/usr/bin/env bash
export SOURCE_ROOT=./

cp -rf ./* $TF_USER_OP_ROOT

cd $TF_USER_OP_ROOT

bazel build -c opt //tensorflow/core/user_ops:siddon_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures

bazel build -c opt //tensorflow/core/user_ops:siddon_tof_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures

bazel build -c opt //tensorflow/core/user_ops:deform_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures

# bazel build -c opt //tensorflow/core/user_ops:scatter_module.so --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures


cd $SOURCE_ROOT
