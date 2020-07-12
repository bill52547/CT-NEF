#load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
#
#tf_custom_op_library(
#    name = "siddon_module.so",
#    srcs = ["siddon/siddon_gpu.cc"],
#    gpu_srcs = ["siddon/siddon_gpu.cu.cc"]
#)
#
#tf_custom_op_library(
#    name = "siddon_tof_module.so",
#    srcs = ["siddon_tof/siddon_tof_gpu.cc"],
#    gpu_srcs = ["siddon_tof/siddon_tof_gpu.cu.cc"]
#)
#
#
#
#tf_custom_op_library(
#    name = "deform_module.so",
#    srcs = ["deform/deform.cc"],
#    gpu_srcs = ["deform/deform.cu.cc"]
#)
#
#
#tf_custom_op_library(
#    name = "scatter_module.so",
#    srcs = ["scatter/scatter_correction.cc"],
#    gpu_srcs = ["scatter/scatter_correction.cu.cc"]
#)
#

package(default_visibility = ["//visibility:public"])
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured", "if_cuda")


cc_library(
    name = "siddon_module_gpu",
    srcs = ["siddon/siddon_gpu.cu.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    alwayslink = 1,
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

cc_binary(
    name = 'python/ops/siddon_module.so',
    srcs = [
        "siddon/siddon_gpu.cc"
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":siddon_module_gpu"]),
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"] + if_cuda(["-DGOOGLE_CUDA=1"])
)



cc_library(
    name = "deform_module_gpu",
    srcs = ["deform/deform.cu.cc"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":cuda",  "@local_config_cuda//cuda:cuda_headers"]),
    alwayslink = 1,
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1", "-x cuda", "-nvcc_options=relaxed-constexpr", "-nvcc_options=ftz=true"]),
)

cc_binary(
    name = 'python/ops/deform_module.so',
    srcs = [
        "deform/deform.cc"
    ],
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":deform_module_gpu"]),
    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"] + if_cuda(["-DGOOGLE_CUDA=1"])
)

