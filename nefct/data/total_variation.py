# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: total_variation.py
@date: 9/1/2019
@desc:
'''
from nefct import nef_class
from nefct.base.base import NefBaseClass
from nefct.data import Image
import numpy as np
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin


@nef_class
class TotalVariation(ArithmeticMixin):
    pass


@nef_class
class TotalVariation2D(TotalVariation):
    data: np.ndarray

    @property
    def x(self):
        return Image(self.data[:, :, 0])

    @property
    def y(self):
        return Image(self.data[:, :, 1])


@nef_class
class TotalVariation3D(TotalVariation):
    data: np.ndarray

    @property
    def x(self):
        return Image(self.data[:, :, :, 0])

    @property
    def y(self):
        return Image(self.data[:, :, :, 1])

    @property
    def z(self):
        return Image(self.data[:, :, :, 2])

    
    def __add__(self, value):
        if not isinstance(value, NefBaseClass):
            return self.update(data = self.data + value)
        else:
            return self.update(data = self.data + value.data)
    def __sub__(self, value):
        if not isinstance(value, NefBaseClass):
            return self.update(data = self.data - value)
        else:
            return self.update(data = self.data - value.data)

@nef_class
class TotalVariation2DT(TotalVariation):
    data: np.ndarray

    @property
    def x(self):
        return Image(self.data[:, :, :, 0])

    @property
    def y(self):
        return Image(self.data[:, :, :, 1])

    @property
    def t(self):
        return Image(self.data[:, :, :, 2])

    def __getitem__(self, item):
        return TotalVariation2D(self.data[:, :, item, :])

@nef_class
class TotalVariation3DTSingle(TotalVariation):
    data: np.ndarray

    @property
    def x(self):
        return Image(self.data[:, :, :, :, 0])

    @property
    def y(self):
        return Image(self.data[:, :, :, :, 1])

    @property
    def z(self):
        return Image(self.data[:, :, :, :, 2])

    @property
    def t(self):
        return Image(self.data[:, :, :, :, 3])


@nef_class
class TotalVariation3DT(TotalVariation):
    data: np.ndarray

    @property
    def x(self):
        return Image(self.data[:, :, :, :, 0])

    @property
    def y(self):
        return Image(self.data[:, :, :, :, 1])

    @property
    def z(self):
        return Image(self.data[:, :, :, :, 2])

    @property
    def t(self):
        return Image(self.data[:, :, :, :, 3])

    def __getitem__(self, item):
        return TotalVariation3DT(self.data[:, :, :, item, :])
