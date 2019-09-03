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
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin
from nefct.ops.common.magic_method_mixins import GetItemMixin
from nefct.io import LoadMixin, SaveMixin
import numpy as np
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin

__all__ = ('Image', 'Image2D', 'Image3D')


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


@nef_class
class Image2D(Image):
    data: np.ndarray
    center: list
    size: list
    timestamp: Any


@nef_class
class Image3D(Image):
    data: np.ndarray
    center: list
    size: list
    timestamp: Any


@nef_class
class Image2DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamp: Any

    def __getitem__(self, item):
        return Image2D(self.data[:, :, item], self.center, self.size, self.timestamp[item])


@nef_class
class Image3DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamp: Any

    def __getitem__(self, item):
        return Image2D(self.data[:, :, :, item], self.center, self.size, self.timestamp[item])
