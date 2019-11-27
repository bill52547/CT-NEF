# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: image.py
@date: 3/20/2019
@desc:
'''

from nefct import nef_class, Any
from nefct.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from nefct.ops.common.imshow_mixin import ImshowMixin
from nefct.ops.common.magic_method_mixins import GetItemMixin
from nefct.io import LoadMixin, SaveMixin
import numpy as np
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin
import attr
from nefct.ops.deform import Deform2DMixin, DeformMixin

__all__ = ('Image', 'Image2D', 'Image3D', 'Image2DT', 'Image3DT', 'all_one_image', 'all_zero_image')


@nef_class
class Image(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
            CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
            LoadMixin, SaveMixin, ImshowMixin, ArithmeticMixin):
    """
    Image data with center and size info.
    """
    data: np.ndarray
    center: list
    size: list
    timestamps: Any


@nef_class
class Image2D(Image, Deform2DMixin):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any

    def deform(self, mx, my):
        if isinstance(mx, Image):
            mx_data = mx.data
        else:
            mx_data = mx

        if isinstance(my, Image):
            my_data = my.data
        else:
            my_data = my

        return self.update(data = self._deform_2d_tf(self.data, mx_data, my_data).numpy())

    def deform_invert(self, mx, my):
        if isinstance(mx, Image):
            mx_data = mx.data
        else:
            mx_data = mx

        if isinstance(my, Image):
            my_data = my.data
        else:
            my_data = my

        return self.update(data = self._deform_invert_2d_tf(self.data, mx_data, my_data).numpy())


@nef_class
class Image3D(Image, DeformMixin):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any

    def deform(self, mx, my, mz):
        if isinstance(mx, Image):
            mx_data = mx.data
        else:
            mx_data = mx

        if isinstance(my, Image):
            my_data = my.data
        else:
            my_data = my

        if isinstance(mz, Image):
            mz_data = mz.data
        else:
            mz_data = mz

        return self.update(data = self._deform_tf(self.data, mx_data, my_data, mz_data).numpy())

    def deform_invert(self, mx, my, mz):
        if isinstance(mx, Image):
            mx_data = mx.data
        else:
            mx_data = mx

        if isinstance(my, Image):
            my_data = my.data
        else:
            my_data = my

        if isinstance(mz, Image):
            mz_data = mz.data
        else:
            mz_data = mz

        new_data = self._deform_invert_tf(self.data, mx_data, my_data, mz_data)
        return self.update(data = new_data.numpy())


@nef_class
class Image2DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any = attr.ib(default = [0])

    def __getitem__(self, item):
        return Image2D(self.data[:, :, item], self.center, self.size, self.timestamps[item])


@nef_class
class Image3DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any = attr.ib(default = [0])

    def __getitem__(self, item):
        return Image3D(self.data[:, :, :, item], self.center, self.size, self.timestamps[item])


def all_one_image(shape: list, timestamps: (list, float) = None):
    if timestamps is None:
        timestamps = 0

    if np.isscalar(timestamps):
        if len(shape) == 2:
            return Image2D(np.ones(shape, np.float32), [0, 0], shape)
        else:
            return Image3D(np.ones(shape, np.float32), [0, 0, 0], shape)

    else:
        nt = len(timestamps)
        shape_ = shape + [nt]
        if len(shape) == 2:
            return Image2DT(np.ones(shape_, np.float32), [0, 0], shape, timestamps = timestamps)
        else:
            return Image3DT(np.ones(shape_, np.float32), [0, 0, 0], shape,
                            timestamps = timestamps)


def all_zero_image(*args, **kwargs):
    return all_one_image(*args, **kwargs) * 0
