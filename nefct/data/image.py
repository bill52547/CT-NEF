# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: image.py
@date: 3/20/2019
@desc:
'''

from nefct import nef_class, List, Any
from nefct.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from nefct.ops.common.imshow_mixin import ImshowMixin
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin
from nefct.ops.common.magic_method_mixins import GetItemMixin
from nefct.io import LoadMixin, SaveMixin

__all__ = ('Image', 'ImageSequence')


@nef_class
class Image(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
            CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
            LoadMixin, SaveMixin, ImshowMixin, ArithmeticMixin):
    """
    Image data with center and size info.
    """

    data: object
    center: list
    size: list


@nef_class
class Image2D(Image):
    data: object  # prefer numpy array
    center: list
    size: list


@nef_class
class Image3D(Image):
    data: object
    center: list
    size: list


@nef_class
class ImageSequence(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
                    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
                    LoadMixin, SaveMixin, ArithmeticMixin):
    """
    Image sequence with center and size info. and timestamp(or surrogate data)
    """

    data: object  # should be list of np.ndarray
    center: list
    size: list
    timestamps: list  # can be number, tuple or a function that return booleans

    def __getitem__(self, ind):
        pass


@nef_class
class ImageSequence2D(ImageSequence):
    data: object  # should be list of np.ndarray
    center: list
    size: list
    timestamps: list  # can be number, tuple or a function that return booleans

    def __getitem__(self, item):
        return Image(self.data[:, :, item], self.center,
                     self.size), self.timestamps[item]


@nef_class
class ImageSequence3D(ImageSequence):
    data: object  # should be list of np.ndarray
    center: list
    size: list
    timestamps: list  # can be number, tuple or a function that return booleans

    def __getitem__(self, item):
        return Image(self.data[:, :, :, item], self.center,
                     self.size), self.timestamps[item]
