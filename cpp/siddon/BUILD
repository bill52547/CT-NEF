#package(default_visibility = ["//visibility:public"])
#load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")
#
#
#
#cc_library(
#    name = "cuda",
#    data = [
#        "@local_config_cuda//cuda:cudart",
#    ],
#    linkopts = select({
#        "@local_config_cuda//cuda:darwin": [
#            "-Wl,-rpath,../local_config_cuda/cuda/lib",
#            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
#        ],
#        "//conditions:default": [
#            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
#            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
#        ],
#    }),
#    deps = [
#        "@local_config_cuda//cuda:cudart",
#    ],
#)
#
#cc_library(
#    name = "siddon_gpu_cu",
#    srcs = ["siddon_gpu.cu.cc"],
#    deps = [
#        "@local_config_tf//:libtensorflow_framework",
#        "@local_config_tf//:tf_header_lib",
#    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
#    alwayslink = 1,
#    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
#)
#
#cc_binary(
#    name = 'siddon_gpu.so',
#    srcs = [
#        "siddon_gpu.cc",
#    ],
#    linkshared = 1,
#    deps = [
#        "@local_config_tf//:libtensorflow_framework",
#        "@local_config_tf//:tf_header_lib",
#    ] + if_cuda_is_configured([":siddon_gpu_cu"]),
#    copts = ["-pthread", "-std=c++11"] + if_cuda(["-DGOOGLE_CUDA=1"])
#)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


tf_custom_op_library(
    name = "siddon_module.so",
    srcs = ["siddon_gpu.cc"],
    gpu_srcs = ["siddon_gpu.cu.cc"]
)
