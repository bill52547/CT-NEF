# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
   name = "new_mat_mod.so",
   srcs = ["new_mat.cc"],
   gpu_srcs = ["new_mat.cu.cc"]
)
