# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_basic_types.py
@date: 4/24/2019
@desc:
'''
import numpy as np

NoneType = type(None)
Any = object


def Optional(type_: type) -> tuple:
    if not isinstance_(type_, type):
        if not type_ == object:
            raise TypeError('argument type_ should be a type, hint:', type_.__name__)
    return (type_, NoneType)


def eq_(type1: type, type2: type) -> bool:
    if issubclass(type1, list) and not issubclass(type2, list):
        return False
    elif issubclass(type1, tuple) and not issubclass(type2, tuple):
        return False
    elif issubclass(type2, list) and not issubclass(type1, list):
        return False
    elif issubclass(type2, tuple) and not issubclass(type1, tuple):
        return False
    elif issubclass(type1, list) and issubclass(type2, list):
        return type1.dtype == type2.dtype and type1.length == type2.length
    elif issubclass(type1, tuple) and issubclass(type2, tuple):
        return type1.dtype == type2.dtype
    else:
        return type1 == type2


# class List(list):
#     __slots__ = ('dtype', 'length')
#
#     def __init__(self, dtype: type = None, length: int = None):
#         self.dtype = dtype
#         self.length = length
#
#     def __call__(self, val):
#         if not isinstance(val, list):
#             raise TypeError(f'argument to make a new `List` instance should be `list`')
#         if self.length is not None:
#             if isinstance(val, (list, tuple)) and not len(val) == self.length:
#                 raise TypeError(
#                     f'argument {val} does not match the required length = {self.length}')
#
#         if self.dtype is not None:
#             if not all(map(lambda x: isinstance(x, self.dtype), val)):
#                 raise TypeError(
#                     f'all the objects in argument should all be type {self.dtype.__name__}')
#         return list(val)
#
#     def __eq__(self, other):
#         if not isinstance(other, List):
#             return False
#         return self.dtype == other.dtype and self.length == other.length
#
#     def isinstance(self, val):
#         if not isinstance(val, list):
#             return False
#         elif self.length is not None and not len(val) == self.length:
#             return False
#         elif self.dtype is not None:
#             return all(map(lambda x: isinstance(x, self.dtype), val))
#         else:
#             return True


def List(dtype: type = None, length: int = None) -> type:
    class NewList(list):
        def __new__(cls, val):
            if not isinstance(val, list):
                raise TypeError('argument to make a new `List` instance should be `list`')

            if length is not None:
                if isinstance(val, (list, tuple)) and not len(val) == length:
                    raise TypeError(
                        'argument', val, 'does not match the required length =', length)

            if dtype is not None:
                if not all(map(lambda x: isinstance(x, dtype), val)):
                    raise TypeError(
                        'all the objects in argument should all be type', {dtype.__name__})

            return super(NewList, cls).__new__(val)

        @classmethod
        def isinstance(cls, val):
            if not isinstance(val, list):
                return False
            elif length is not None and not len(val) == length:
                return False
            elif dtype is not None:
                return all(map(lambda x: isinstance(x, dtype), val))
            else:
                return True

    NewList.dtype = dtype
    NewList.length = length
    NewList.__name__ = 'List(' + dtype.__name__ + ',' + str(length)

    return NewList


def Tuple(dtype: tuple = None) -> type:
    if not all(map(lambda x: isinstance(x, type), dtype)):
        raise TypeError('`Tuple` generator arguments should be a `list` of `types`')

    class NewTuple(tuple):
        def __new__(cls, val):
            if not isinstance(val, tuple):
                raise TypeError('argument to make a new `Tuple` instance should be `tuple`')

            if dtype is not None:
                if not len(val) == len(dtype):
                    raise TypeError('argument length does not match the dtype tuple length,',
                                    'which is', len(dtype))
                for i, (val_, dtype_) in enumerate(zip(val, dtype)):
                    if not isinstance(val_, dtype_):
                        raise TypeError('argument', i, 'does not match the type requirement, ',
                                        'which is', dtype_.__name__)

            return super(NewTuple, cls).__new__(cls, val)

        @classmethod
        def isinstance(cls, val):
            if not isinstance(val, tuple):
                return False
            elif dtype is not None:
                return all(map(lambda x: isinstance(x[0], x[1]), zip(val, dtype)))
            else:
                return True

    NewTuple.dtype = dtype
    NewTuple.__name__ = 'Tuple(' + str([type_.__name__ for type_ in dtype])

    return NewTuple


def isinstance_(x: Any, A_tuple: tuple) -> bool:
    if isinstance(A_tuple, type):
        A_tuple = tuple([A_tuple])
    for type_ in A_tuple:
        isinstance_func = getattr(type_, 'isinstance', None)
        if isinstance_func is not None:
            if type_.isinstance(x):
                return True
        else:
            if type_ is int:
                type_ = (int, np.int, np.int64)
            if isinstance(x, type_):
                return True

    return False
