# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
   name = "tf_test.so",
   srcs = ["mul_two.cc"],
   gpu_srcs = ["mul_two.cu.cc"]
)
