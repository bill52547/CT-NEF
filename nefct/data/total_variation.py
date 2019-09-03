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
        return TotalVariation3D(self.data[:, :, :, item, :])
