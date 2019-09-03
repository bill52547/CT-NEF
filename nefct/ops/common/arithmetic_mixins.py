# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: arithmetic_mixins.py
@date: 5/8/2019
@desc:
'''


class NegMixin:
    def __neg__(self):
        return self.update(data = -self.data)


class AbsMixin:
    def abs(self):
        return self.update(data = abs(self.data))


class AddMixin:
    def __add__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data + other.data)
        else:
            return self.update(data = self.data + other)


class SubMixin:
    def __sub__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data - other.data)
        else:
            return self.update(data = self.data - other)


class MulMixin:
    def __mul__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data * other.data)
        else:
            return self.update(data = self.data * other)


class TrueDivMixin:
    def __truediv__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data / other.data)
        else:
            return self.update(data = self.data / other)


class FloorDivMixin:
    def __floordiv__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data // other.data)
        else:
            return self.update(data = self.data // other)


class ModMixin:
    def __mod__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data % other.data)
        else:
            return self.update(data = self.data % other)


class PowMixin:
    def __pow__(self, other):
        if hasattr(other, 'data'):
            return self.update(data = self.data ** other.data)
        else:
            return self.update(data = self.data ** other)


class ArithmeticMixin(NegMixin,
                      AbsMixin,
                      AddMixin,
                      SubMixin,
                      MulMixin,
                      TrueDivMixin,
                      FloorDivMixin,
                      ModMixin,
                      PowMixin):
    pass
