# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_base.py
@date: 4/24/2019
@desc:
'''
import inspect
from abc import abstractmethod
# from srfnef import __version__
import attr
import types
import numpy as np
from .basic_types import Any, Optional, isinstance_

__all__ = ('NefBaseClass', 'nef_class')

method_name_exceptions = ['keys', 'types', 'update', 'asdict', 'values', 'items', 'abs',
                          'func_keys', 'func_types', 'func_signatures', 'astype', 'from_dict']


class BaseMeta(type):
    def __new__(cls, *args, **kwargs):
        return super(BaseMeta, cls).__new__(cls, *args, **kwargs)


class NefBaseClass:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '__annotations__'):
            return super(NefBaseClass, cls).__new__(cls)
        return super(NefBaseClass, cls).__new__(cls)

    @classmethod
    def keys(cls) -> list:
        return list(getattr(cls, '__annotations__', {}).keys())

    def values(self) -> Any:
        return [getattr(self, key) for key in self.keys()]

    def items(self, recurse: bool = False) -> list:
        return list(attr.asdict(self, recurse=recurse).items())

    @classmethod
    def types(cls) -> list:
        return list(getattr(cls, '__annotations__', {}).values())

    def update(self, **kwargs) -> 'NefBaseClass':
        return attr.evolve(self, **kwargs)

    def asdict(self, recurse=False) -> dict:
        return attr.asdict(self, recurse=recurse)

    @classmethod
    def from_dict(cls, dct: dict) -> object:
        attr_dict = {}
        if dct is None:
            return cls()
        for key, type_ in cls.__annotations__.items():
            if key not in dct:
                continue
            if dct[key] is None:
                attr_dict.update({key: None})
            elif not isinstance(dct[key], dict):
                attr_dict.update({key: dct[key]})
            elif issubclass(type_, NefBaseClass):
                attr_dict.update({key: type_.from_dict(dct[key])})
            else:
                val = dct[key]
                if isinstance(val, np.bool_):
                    val = bool(val)
                attr_dict.update({key: val})

        return cls(**attr_dict)

    def astype(self, cls: type) -> object:
        return cls.from_dict(self.asdict(recurse=True))

    @abstractmethod
    def __attrs_post_init__(self) -> None:
        pass

    def __call__(self, *args, **kwargs):
        return self

    def attr_eq(self, other) -> bool:
        if not isinstance(other, NefBaseClass):
            return False
        if hasattr(self, 'data'):
            return self.update(data=0).asdict() == other.update(data=0).asdict()
        else:
            return self.asdict() == other.asdict()

    @classmethod
    def __call_annotations__(cls) -> dict:
        sig = inspect.signature(cls.__call__)
        dct = {}
        for name, type_ in sig.parameters.items():
            if name == 'self':
                continue
            dct.update({name: type_.annotation.__name__})
        dct.update({'return': sig.return_annotation.__name__})
        return dct


def nef_class(cls) -> Any:
    attr_cls = attr.s(auto_attribs=True)(cls)
    args_dict = {}
    for key, val in attr.fields_dict(attr_cls).items():
        if isinstance(val.default, attr._make._Nothing):
            _default = None
        else:
            _default = val.default

        if not isinstance_(val.type, tuple):
            type_ = Optional(val.type)
        else:
            type_ = val.type
        args_dict.update({key: attr.ib(type=type_, default=_default)})

    new_cls = attr.make_class(cls.__name__,
                              args_dict,
                              bases=(cls,),
                              auto_attribs=True,
                              # frozen = True,
                              slots=True)

    return types.new_class(cls.__name__, (new_cls, NefBaseClass),
                           {'metaclass': BaseMeta})
