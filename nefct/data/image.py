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

__all__ = ('Image', )


@nef_class
class Image(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
            CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
            LoadMixin, SaveMixin, ImshowMixin, ArithmeticMixin):
    data: np.ndarray
    center: list
    size: list
    timestamps: Any
