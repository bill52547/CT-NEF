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

__all__ = ('Image', )


@nef_class
class Image(ShapePropertyMixin, UnitSizePropertyMixin, GetItemMixin,
            CentralSlicesPropertyMixin, CentralProfilesPropertyMixin,
            LoadMixin, SaveMixin, ImshowMixin, ArithmeticMixin, DeformMixin):
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

        return self.update(data=self._deform_tf(self.data, mx_data, my_data, mz_data).numpy())

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
        return self.update(data=new_data.numpy())
