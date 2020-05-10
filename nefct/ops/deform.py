# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: deform_mixin.py
@date: 5/6/2019
@desc:
'''
from nefct import nef_class
import tensorflow as tf
from nefct.config import TF_USER_OP_PATH

# deform_tex_module = tf.load_op_library(
#     TF_USER_OP_PATH + '/tf_deform_tex_module.so'
# )

# deform_2d_tex_module = tf.load_op_library(
#     TF_USER_OP_PATH + '/tf_deform_2d_tex_module.so'
# )


@nef_class
class DeformMixin:
    def _deform_tf(self, image_data: tf.Tensor,
                   mx: tf.Tensor,
                   my: tf.Tensor,
                   mz: tf.Tensor) -> tf.Tensor:
        image_shape = [int(s) for s in image_data.shape]
        if isinstance(image_data, tf.Variable):
            image_data_tf = tf.transpose(image_data)
        else:
            image_data_tf = tf.transpose(tf.constant(image_data))
        if isinstance(mx, tf.Tensor):
            mx_tf = tf.transpose(mx)
        else:
            mx_tf = tf.transpose(tf.constant(mx))
        if isinstance(my, tf.Tensor):
            my_tf = tf.transpose(my)
        else:
            my_tf = tf.transpose(tf.constant(my))
        if isinstance(mz, tf.Tensor):
            mz_tf = tf.transpose(mz)
        else:
            mz_tf = tf.transpose(tf.constant(mz))
        return tf.transpose(
            deform_tex_module.deform(image_data_tf, mx_tf, my_tf, mz_tf, image_shape))

    def _deform_invert_tf(self, image_data: tf.Tensor,
                          mx: tf.Tensor,
                          my: tf.Tensor,
                          mz: tf.Tensor) -> tf.Tensor:
        image_shape = [int(s) for s in image_data.shape]
        if isinstance(image_data, tf.Variable):
            image_data_tf = tf.transpose(image_data)
        else:
            image_data_tf = tf.transpose(tf.constant(image_data))
        if isinstance(mx, tf.Tensor):
            mx_tf = tf.transpose(mx)
        else:
            mx_tf = tf.transpose(tf.constant(mx))
        if isinstance(my, tf.Tensor):
            my_tf = tf.transpose(my)
        else:
            my_tf = tf.transpose(tf.constant(my))
        if isinstance(mz, tf.Tensor):
            mz_tf = tf.transpose(mz)
        else:
            mz_tf = tf.transpose(tf.constant(mz))
        return tf.transpose(deform_tex_module.deform_invert(image_data_tf,
                                                            mx_tf, my_tf, mz_tf,
                                                            image_shape))


# @nef_class
# class Deform2DMixin:
#     def _deform_2d_tf(self, image_data: tf.Tensor,
#                       mx: tf.Tensor,
#                       my: tf.Tensor) -> tf.Tensor:
#         image_shape = [int(s) for s in image_data.shape]
#         if isinstance(image_data, tf.Variable):
#             image_data_tf = tf.transpose(image_data)
#         else:
#             image_data_tf = tf.transpose(tf.constant(image_data))
#         if isinstance(mx, tf.Tensor):
#             mx_tf = tf.transpose(mx)
#         else:
#             mx_tf = tf.transpose(tf.constant(mx))
#         if isinstance(my, tf.Tensor):
#             my_tf = tf.transpose(my)
#         else:
#             my_tf = tf.transpose(tf.constant(my))
#         return tf.transpose(
#             deform_2d_tex_module.deform_two(image_data_tf, mx_tf, my_tf, image_shape))

#     def _deform_invert_2d_tf(self, image_data: tf.Tensor,
#                              mx: tf.Tensor,
#                              my: tf.Tensor) -> tf.Tensor:
#         image_shape = [int(s) for s in image_data.shape]
#         if isinstance(image_data, tf.Variable):
#             image_data_tf = tf.transpose(image_data)
#         else:
#             image_data_tf = tf.transpose(tf.constant(image_data))
#         if isinstance(mx, tf.Tensor):
#             mx_tf = tf.transpose(mx)
#         else:
#             mx_tf = tf.transpose(tf.constant(mx))
#         if isinstance(my, tf.Tensor):
#             my_tf = tf.transpose(my)
#         else:
#             my_tf = tf.transpose(tf.constant(my))
#         return tf.transpose(deform_2d_tex_module.deform_invert_two(image_data_tf,
#                                                                    mx_tf, my_tf,
#                                                                    image_shape))
