# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: local_io_mixin.py
@date: 4/30/2019
@desc:
'''
import deepdish as dd
from nefct import NefBaseClass


class SaveMixin:
    def _save(self, path: str = None) -> str:
        if path is None:
            from nefct.geometry import RESOURCE_DIR
            from nefct.utils import get_hash_of_timestamp
            path = RESOURCE_DIR + get_hash_of_timestamp() + '.hdf5'
        dct = self.asdict(recurse = True)
        dd.io.save(path, dct, compression = None)

        return path

    def save(self, path: str = None) -> str:
        return self._save(path)


class LoadMixin:
    @classmethod
    def _load(cls, path: str = None, partial = None):
        if path is None:
            raise ValueError('loading path is necessary')
        if partial is None:
            dct = dd.io.load(path)
        else:
            dct = dd.io.load(path, partial)
        return cls.from_dict(dct)

    @classmethod
    def load(cls, path: str = None, partial = None):
        return cls._load(path, partial)


def save(obj: NefBaseClass, path: str = None) -> str:
    if path is None:
        from nefct.geometry import RESOURCE_DIR
        from nefct.utils import get_hash_of_timestamp
        path = RESOURCE_DIR + get_hash_of_timestamp() + '.hdf5'
    dct = obj.asdict(recurse = True)
    dd.io.save(path, dct, compression = None)
    return path


def load(cls: type, path: str, partial = None) -> NefBaseClass:
    assert issubclass(cls, NefBaseClass)
    if partial is None:
        dct = dd.io.load(path)
    else:
        dct = dd.io.load(path, partial)
    return cls.from_dict(dct)
