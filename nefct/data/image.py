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

__all__ = ('Image', 'Image2D', 'Image3D', 'Image2DT', 'Image3DT', 'all_one_image', 'all_zero_image')


@nef_class
class Image(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
            CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
            LoadMixin, SaveMixin, ImshowMixin, ArithmeticMixin):
    """
    Image data with center and size info.
    """
    pass

@nef_class
class Image2D(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any


@nef_class
class Image3D(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any


@nef_class
class Image2DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any

    def __getitem__(self, item):
        return Image2D(self.data[:, :, item], self.center, self.size, self.timestamps[item])


@nef_class
class Image3DT(Image):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any

    def __getitem__(self, item):
        return Image2D(self.data[:, :, :, item], self.center, self.size, self.timestamps[item])

def all_one_image(shape:list, timestamps: (list, float) = None):
    if timestamps is None:
        timestamps = 0

    if np.isscalar(timestamps):
        if len(shape) == 2:
            return Image2D(np.ones(shape, np.float32), [0, 0], shape)
        else:
            return Image3D(np.ones(shape, np.float32), [0,0,0], shape)
    
    else:
        nt = len(timestamps)
        shape_ = shape + [nt]
        if len(shape) == 2:
            return Image2DT(np.ones(shape_, np.float32), [0,0], shape, timestamps = timestamps)
        else:
            return Image3DT(np.ones(shape_, np.float32), [0,0,0], shape, timestamps = timestamps)
        

def all_zero_image(*args, **kwargs):
    return all_one_image(*args, **kwargs)