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
from nefct.ops.deform import DeformMixin

__all__ = ('Image', 'Image2D', 'Image3D', 'Image2DT', 'Image3DT')


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
class Image2D(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any

    # def deform(self, mx, my):
    #     if isinstance(mx, Image):
    #         mx_data = mx.data
    #     else:
    #         mx_data = mx

    #     if isinstance(my, Image):
    #         my_data = my.data
    #     else:
    #         my_data = my

    #     return self.update(data = self._deform_2d_tf(self.data, mx_data, my_data).numpy())

    # def deform_invert(self, mx, my):
    #     if isinstance(mx, Image):
    #         mx_data = mx.data
    #     else:
    #         mx_data = mx

    #     if isinstance(my, Image):
    #         my_data = my.data
    #     else:
    #         my_data = my

    #     return self.update(data = self._deform_invert_2d_tf(self.data, mx_data, my_data).numpy())


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
    timestamps: Any = attr.ib(default = [0] * 12)

    def __getitem__(self, item):
        return Image3D(self.data[:, :, :, item], self.center, self.size, self.timestamps[item])

    def __len__(self):
        return self.data.shape[3]