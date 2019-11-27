# !/bin/sh
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

# tf_custom_op_library(
#    name = "ct_siddon2d_module.so",
#    srcs = ["siddon_2d/siddon.cc"],
#    gpu_srcs = ["siddon_2d/siddon.cu.cc"]
# )

# tf_custom_op_library(
#    name = "ct_siddon3d_module.so",
#    srcs = ["siddon_3d/siddon.cc"],
#    gpu_srcs = ["siddon_3d/siddon.cu.cc"]
# )

# tf_custom_op_library(
#    name = "ct_ray2d_module.so",
#    srcs = ["ray_2d/ray.cc"],
#    gpu_srcs = ["ray_2d/ray.cu.cc"]
# )

# tf_custom_op_library(
#    name = "ct_ray3d_module.so",
#    srcs = ["ray_3d/ray.cc"],
#    gpu_srcs = ["ray_3d/ray.cu.cc"]
# )


# tf_custom_op_library(
#    name = "ct_pixel2d_module.so",
#    srcs = ["pixel_2d/pixel.cc"],
#    gpu_srcs = ["pixel_2d/pixel.cu.cc"]
# )

# tf_custom_op_library(
#    name = "ct_pixel3d_module.so",
#    srcs = ["pixel_3d/pixel.cc"],
#    gpu_srcs = ["pixel_3d/pixel.cu.cc"]
# )

tf_custom_op_library(
   name = "dist_3d_mod.so",
   srcs = ["distance_3d/dist_3d.cc"],
   gpu_srcs = ["distance_3d/dist_3d.cu.cc", "distance_3d/dist_3d_back.cu.cc", "distance_3d/dist_3d_back_fdk.cu.cc"]
)

# tf_custom_op_library(
#    name = "dist_3d_deform_mod.so",
#    srcs = ["distance_deform_3d/dist_3d.cc"],
#    gpu_srcs = ["distance_deform_3d/dist_3d.cu.cc", "distance_deform_3d/dist_3d_back.cu.cc"]
# )


# tf_custom_op_library(
#    name = "dist_2d_mod.so",
#    srcs = ["distance_2d/dist_2d.cc"],
#    gpu_srcs = ["distance_2d/dist_2d.cu.cc", "distance_2d/dist_2d_back.cu.cc"]
# )
# tf_custom_op_library(
#   name = "tf_deform_tex_module.so",
#   srcs = ["deform/deform_tex.cc"],
#   gpu_srcs = ["deform/deform_tex.cu.cc"]
# )

# tf_custom_op_library(
#   name = "tf_deform_2d_tex_module.so",
#   srcs = ["deform/deform_2d_tex.cc"],
#   gpu_srcs = ["deform/deform_2d_tex.cu.cc"]
# )

tf_custom_op_library(
  name = "tf_affine_module.so",
  srcs = ["affine_tex/affine_deform_tex.cc"],
  gpu_srcs = ["affine_tex/affine_deform_tex.cu.cc"]
)

tf_custom_op_library(
  name = "generate_sys_mat_mod.so",
  srcs = ["gen_sys_mat/generate_sys_mat.cc"],
  gpu_srcs = ["gen_sys_mat/count_sys_mat.cu.cc","gen_sys_mat/generate_sys_mat.cu.cc"]
)


tf_custom_op_library(
  name = "ssm_op_mod.so",
  srcs = ["ssm_ops/ssm_ops.cc"],
  gpu_srcs = ["ssm_ops/ssm_ops.cu.cc"]
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
