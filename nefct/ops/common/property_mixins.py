# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: basenef
@file: arithematic_ops.py
@date: 4/13/2019
@desc:
'''
__all__ = ('ShapePropertyMixin', 'NdimPropertyMixin', 'LengthPropertyMixin',
           'CentralProfilesPropertyMixin',
           'CentralSlicesPropertyMixin', 'UnitSizePropertyMixin')


class ShapePropertyMixin:
    @property
    def shape(self):
        return [int(x) for x in self.data.shape]


class NdimPropertyMixin(ShapePropertyMixin):
    @property
    def ndim(self):
        return len(self.shape)


class LengthPropertyMixin:
    def __len__(self):
        return int(self.data.shape[0])

    @property
    def length(self):
        return int(self.data.shape[0])


class CentralSlicesPropertyMixin:
    @property
    def central_slices(self):
        t0 = self.data[int(self.shape[0] / 2), :, :]
        t1 = self.data[:, int(self.shape[1] / 2), :]
        t2 = self.data[:, :, int(self.shape[2] / 2)]
        return t0, t1, t2


class CentralProfilesPropertyMixin:
    @property
    def central_profiles(self):
        p0 = self.data[:, int(self.shape[1] / 2), int(self.shape[2] / 2)]
        p1 = self.data[int(self.shape[0] / 2), :, int(self.shape[2] / 2)]
        p2 = self.data[int(self.shape[0] / 2), int(self.shape[1] / 2), :]
        return p0, p1, p2


class UnitSizePropertyMixin:
    @property
    def unit_size(self):
        return [s1 / s2 for s1, s2 in zip(self.size, self.shape)]
