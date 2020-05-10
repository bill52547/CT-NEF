# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
   name = "tf_add_one.so",
   srcs = ["add_one.cc"],
   gpu_srcs = ["add_one.cu.cc"]
)
