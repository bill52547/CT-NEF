# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: total_variation.py
@date: 8/28/2019
@desc:
'''

from nefct import nef_class
from nefct.data import Image
from nefct.data.image import Image2DT, Image3DT
from nefct.data.total_variation import TotalVariation2D, TotalVariation2DT, TotalVariation3D, \
    TotalVariation3DT, TotalVariation
import numpy as np


@nef_class
class TotalVari:
    def __call__(self, image: Image) -> TotalVariation:
        if len(image.shape) == 2:
            tv_x_data = image.data * 0
            tv_y_data = image.data * 0
            tv_x_data[1:, :] = image.data[1:, :] - image.data[:-1, :]
            tv_y_data[:, 1:] = image.data[:, 1:] - image.data[:, :-1]
            return TotalVariation2D(np.stack([tv_x_data, tv_y_data], axis = 2))
        else:
            tv_x_data = image.data * 0
            tv_y_data = image.data * 0
            tv_z_data = image.data * 0
            tv_x_data[1:, :, :] = image.data[1:, :, :] - image.data[:-1, :, :]
            tv_y_data[:, 1:, :] = image.data[:, 1:, :] - image.data[:, :-1, :]
            tv_z_data[:, :, 1:] = image.data[:, :, 1:] - image.data[:, :, :-1]
            return TotalVariation3D(np.stack([tv_x_data, tv_y_data, tv_z_data], axis = 3))


#
# @nef_class
# class TotalVariIso:
#     def __call__(self, image: Image) -> TotalVariation:
#         if len(image.shape) == 2:
#             tv_x = image.update(data = image.data * 0)
#             tv_y = image.update(data = image.data * 0)
#             tv_x.data[1:, :] = image.data[1:, :] - image.data[:-1, :]
#             tv_y.data[:, 1:] = image.data[:, 1:] - image.data[:, :-1]
#             iso_val = np.sqrt(tv_x.data ** 2 + tv_y.data ** 2)
#             return image.update(data = iso_val)
#         else:
#             tv_x = image.update(data = image.data * 0)
#             tv_y = image.update(data = image.data * 0)
#             tv_z = image.update(data = image.data * 0)
#             tv_x.data[1:, :, :] = image.data[1:, :, :] - image.data[:-1, :, :]
#             tv_y.data[:, 1:, :] = image.data[:, 1:, :] - image.data[:, :-1, :]
#             tv_z.data[:, :, 1:] = image.data[:, :, 1:] - image.data[:, :, :-1]
#             iso_val = np.sqrt(tv_x.data ** 2 + tv_y.data ** 2 + tv_z.data ** 2)
#             return image.update(data = iso_val)
#

@nef_class
class TotalVariT:
    def __call__(self, image: Image) -> TotalVariation:
        if isinstance(image, Image2DT):
            image_array = image.data
            tv_x_data = image_array * 0
            tv_y_data = image_array * 0
            tv_t_data = image_array * 0
            tv_x_data[1:, :, :] = image_array[1:, :, :] - image_array[:-1, :, :]
            tv_y_data[:, 1:, :] = image_array[:, 1:, :] - image_array[:, :-1, :]
            tv_t_data[:, :, 1:] = image_array[:, :, 1:] - image_array[:, :, :-1]
            return TotalVariation2DT(np.stack((tv_x_data, tv_y_data, tv_t_data), axis = 3))
        else:
            image_array = image.data
            tv_x_data = image_array * 0
            tv_y_data = image_array * 0
            tv_z_data = image_array * 0
            tv_t_data = image_array * 0
            tv_x_data[1:, :, :, :] = image_array[1:, :, :, :] - image_array[:-1, :, :, :]
            tv_y_data[:, 1:, :, :] = image_array[:, 1:, :, :] - image_array[:, :-1, :, :]
            tv_z_data[:, :, 1:, :] = image_array[:, :, 1:, :] - image_array[:, :, :-1, :]
            tv_t_data[:, :, :, 1:] = image_array[:, :, :, 1:] - image_array[:, :, :, :-1]
            return TotalVariation3DT(np.stack((tv_x_data, tv_y_data, tv_t_data, tv_t_data),
                                              axis = 4))


# @nef_class
# class TotalVariTIso:
#     def __call__(self, images: list) -> TotalVariation:
#         tv_x_list = []
#         tv_y_list = []
#         tv_z_list = []
#         tv_t_list = []
#
#         if len(images[0].shape) == 2:
#             for image in images:
#                 tv_x = image.update(data = image.data * 0)
#                 tv_y = image.update(data = image.data * 0)
#                 tv_x.data[1:, :] = image.data[1:, :] - image.data[:-1, :]
#                 tv_y.data[:, 1:] = image.data[:, 1:] - image.data[:, :-1]
#                 tv_x_list.append(tv_x)
#                 tv_y_list.append(tv_y)
#             tv_t_list.append(images[0] * 0)
#             for img1, img2 in zip(images[:-1], images[1:]):
#                 tv_t_list.append(img2 - img1)
#
#             return TotalVariation2DT(tv_x_list, tv_y_list, tv_t_list)
#         else:
#             for image in images:
#                 tv_x = image.update(data = image.data * 0)
#                 tv_y = image.update(data = image.data * 0)
#                 tv_z = image.update(data = image.data * 0)
#                 tv_x.data[1:, :, :] = image.data[1:, :, :] - image.data[:-1, :, :]
#                 tv_y.data[:, 1:, :] = image.data[:, 1:, :] - image.data[:, :-1, :]
#                 tv_z.data[:, :, 1:] = image.data[:, :, 1:] - image.data[:, :, :-1]
#                 tv_x_list.append(tv_x)
#                 tv_y_list.append(tv_y)
#                 tv_z_list.append(tv_z)
#
#             tv_t_list.append(images[0] * 0)
#             for img1, img2 in zip(images[:-1], images[1:]):
#                 tv_t_list.append(img2 - img1)
#             return TotalVariation3DT(tv_x_list, tv_y_list, tv_z_list, tv_t_list)
#

@nef_class
class InvertTotalVari:
    def __call__(self, tv_val: TotalVariation) -> (Image, list):

        if isinstance(tv_val, TotalVariation2D):
            tv_x, tv_y = tv_val.x, tv_val.y
            itv_x = tv_x * 0
            itv_y = tv_y * 0
            itv_x.data[:-1, :] = -tv_x.data[1:, :]
            itv_x.data[1:, :] += tv_x.data[1:, :]
            itv_y.data[:, :-1] = -tv_y.data[:, 1:]
            itv_y.data[:, 1:] += tv_y.data[:, 1:]
            return itv_x + itv_y
        elif isinstance(tv_val, TotalVariation3D):
            tv_x, tv_y, tv_z = tv_val.x, tv_val.y, tv_val.z
            itv_x = tv_x * 0
            itv_y = tv_y * 0
            itv_z = tv_z * 0
            itv_x.data[:-1, :, :] = -tv_x.data[1:, :, :]
            itv_x.data[1:, :, :] += tv_x.data[1:, :, :]
            itv_y.data[:, :-1, :] = -tv_y.data[:, 1:, :]
            itv_y.data[:, 1:, :] += tv_y.data[:, 1:, :]
            itv_z.data[:, :, :-1] = -tv_z.data[:, :, 1:]
            itv_z.data[:, :, 1:] += tv_z.data[:, :, 1:]
            return itv_x + itv_y + itv_z
        elif isinstance(tv_val, TotalVariation2DT):
            tv_x, tv_y, tv_t = tv_val.x, tv_val.y, tv_val.t
            itv_x = tv_x * 0
            itv_y = tv_y * 0
            itv_t = tv_t * 0
            itv_x.data[:-1, :, :] = -tv_x.data[1:, :, :]
            itv_x.data[1:, :, :] += tv_x.data[1:, :, :]
            itv_y.data[:, :-1, :] = -tv_y.data[:, 1:, :]
            itv_y.data[:, 1:, :] += tv_y.data[:, 1:, :]
            itv_t.data[:, :, :-1] = -tv_t.data[:, :, 1:]
            itv_t.data[:, :, 1:] += tv_t.data[:, :, 1:]
            image = tv_val.x * 0
            for i in range(tv_x.shape[-1]):
                image.data[:, :, i] += itv_x.data[:, :, i] + itv_y.data[:, :, i] + itv_t.data[:, :,
                                                                                   i]
            return image

        elif isinstance(tv_val, TotalVariation3DT):
            tv_x, tv_y, tv_z, tv_t = tv_val.x, tv_val.y, tv_val.z, tv_val.t
            itv_x = tv_x * 0
            itv_y = tv_y * 0
            itv_z = tv_z * 0
            itv_t = tv_t * 0
            itv_x.data[:-1, :, :, :] = -tv_x.data[1:, :, :, :]
            itv_x.data[1:, :, :, :] += tv_x.data[1:, :, :, :]
            itv_y.data[:, :-1, :, :] = -tv_y.data[:, 1:, :, :]
            itv_y.data[:, 1:, :, :] += tv_y.data[:, 1:, :, :]
            itv_z.data[:, :, :-1, :] = -tv_z.data[:, :, 1:, :]
            itv_z.data[:, :, 1:, :] += tv_z.data[:, :, 1:, :]
            itv_t.data[:, :, :, :-1] = -tv_z.data[:, :, :, 1:]
            itv_t.data[:, :, :, 1:] += tv_z.data[:, :, :, 1:]
            image = tv_val.x * 0
            for i in range(tv_x.shape[-1]):
                image.data += itv_x[:, :, :, i] + itv_y[:, :, :, i] + itv_z[:, :, :, i] + itv_t[:,
                                                                                          :, :, i]
            return image

        else:
            raise NotImplementedError

#
# @nef_class
# class InvertTotalVariT:
#     def __call__(self, TotalVariation) -> list:
#         if len(args) == 3:
#             tv_x_list, tv_y_list, tv_t_list = args
#             nx, ny = tv_t_list[0].shape
#             nt = len(tv_t_list)
#
#             itv_x_data = np.zeros((nx, ny, nt), np.float32)
#             itv_y_data = np.zeros((nx, ny, nt), np.float32)
#             itv_t_data = np.zeros((nx, ny, nt), np.float32)
#
#             for i in range(nt):
#                 itv_x_data[:-1, :, i] = -tv_x_list[i].data[1:, :]
#                 itv_x_data[1:, :, i] += tv_x_list[i].data[1:, :]
#                 itv_y_data[:, :-1, i] = -tv_y_list[i].data[:, 1:]
#                 itv_y_data[:, 1:, i] += tv_y_list[i].data[:, 1:]
#                 if not i == nt - 1:
#                     itv_t_data[:, :, i] = -tv_t_list[i + 1].data[:, :]
#                     itv_t_data[:, :, i + 1] += tv_t_list[i + 1].data[:, :]
#             sum_itv = itv_x_data + itv_y_data + itv_t_data
#             return [tv_x_list[0].update(data = sum_itv[:, :, i]) for i in range(nt)]
#
#         else:
#             tv_x_list, tv_y_list, tv_z_list, tv_t_list = args
#             nx, ny, nz = tv_t_list[0].shape
#             nt = len(tv_t_list)
#
#             itv_x_data = np.zeros((nx, ny, nz, nt), np.float32)
#             itv_y_data = np.zeros((nx, ny, nz, nt), np.float32)
#             itv_z_data = np.zeros((nx, ny, nz, nt), np.float32)
#             itv_t_data = np.zeros((nx, ny, nz, nt), np.float32)
#
#             for i in range(nt):
#                 itv_x_data[:-1, :, :, i] = -tv_x_list[i].data[1:, :, :]
#                 itv_x_data[1:, :, :, i] += tv_x_list[i].data[1:, :, :]
#                 itv_y_data[:, :-1, :, i] = -tv_y_list[i].data[:, 1:, :]
#                 itv_y_data[:, 1:, :, i] += tv_y_list[i].data[:, 1:, :]
#                 itv_z_data[:, :, :-1, i] = -tv_z_list[i].data[:, :, 1:]
#                 itv_z_data[:, :, 1:, i] += tv_z_list[i].data[:, :, 1:]
#                 if not i == nt - 1:
#                     itv_t_data[:, :, :, i] = -tv_t_list[i + 1].data
#                     itv_t_data[:, :, :, i + 1] += tv_t_list[i + 1].data
#             sum_itv = itv_x_data + itv_y_data + itv_z_data + itv_t_data
#             return [tv_x_list[0].update(data = sum_itv[:, :, :, i]) for i in range(nt)]
