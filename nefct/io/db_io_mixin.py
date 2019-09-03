# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: local_io_mixin.py
@date: 4/30/2019
@desc:
'''
from nefct import NefBaseClass, nef_class
import deepdish as dd


@nef_class
class SaveMixin:
    def _save(self, x: NefBaseClass, path: str = None) -> str:
        if path is None:
            from nefct.geometry import RESOURCE_DIR
            from nefct.utils import get_hash_of_timestamp
            path = RESOURCE_DIR + get_hash_of_timestamp() + '.hdf5'
        dct = x.asdict(recurse = True)
        dd.io.save(path, dct, compression = None)
        return path

    def save(self, x: NefBaseClass, path: str = None) -> str:
        return self._save(x, path)


@nef_class
class LoadMixin:
    def _load(self, path: str = None, cls: type = None) -> NefBaseClass:
        if path is None:
            raise ValueError('loading path is necessary')
        dct = dd.io.load(path)
        if cls is None:
            return dct
        else:
            return cls(**dct)

    def load(self, path: str = None, cls: type = None) -> NefBaseClass:
        return self._load(path, cls)
