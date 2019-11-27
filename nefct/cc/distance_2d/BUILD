# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
   name = "dist_2d_mod.so",
   srcs = ["dist_2d.cc"],
   gpu_srcs = ["dist_2d.cu.cc", "dist_2d_back.cu.cc"]
)
