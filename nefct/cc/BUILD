# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
   name = "ct_siddon2d_module.so",
   srcs = ["siddon_2d/siddon.cc"],
   gpu_srcs = ["siddon_2d/siddon.cu.cc"]
)

tf_custom_op_library(
   name = "ct_siddon3d_module.so",
   srcs = ["siddon_3d/siddon.cc"],
   gpu_srcs = ["siddon_3d/siddon.cu.cc"]
)

tf_custom_op_library(
   name = "ct_ray2d_module.so",
   srcs = ["ray_2d/ray.cc"],
   gpu_srcs = ["ray_2d/ray.cu.cc"]
)

tf_custom_op_library(
   name = "ct_ray3d_module.so",
   srcs = ["ray_3d/ray.cc"],
   gpu_srcs = ["ray_3d/ray.cu.cc"]
)


tf_custom_op_library(
   name = "ct_pixel2d_module.so",
   srcs = ["pixel_2d/pixel.cc"],
   gpu_srcs = ["pixel_2d/pixel.cu.cc"]
)

tf_custom_op_library(
   name = "ct_pixel3d_module.so",
   srcs = ["pixel_3d/pixel.cc"],
   gpu_srcs = ["pixel_3d/pixel.cu.cc"]
)

# tf_custom_op_library(
#    name = "tf_siddon_module.so",
#    srcs = ["siddon/siddon.cc"],
#    gpu_srcs = ["siddon/siddon.cu.cc"]
# )


# tf_custom_op_library(
#    name = "tf_deform_module.so",
#    srcs = ["deform/deform.cc"],
#    gpu_srcs = ["deform/deform.cu.cc"]
# )
