# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: data_io_mixin.py
@date: 5/9/2019
@desc:
'''
import deepdish as dd
from nefct import NefBaseClass
from nefct.config import RESOURCE_DIR
from nefct.utils import get_hash_of_timestamp


def dump_data(obj: NefBaseClass) -> NefBaseClass:
    path = RESOURCE_DIR + get_hash_of_timestamp() + '.hdf5'
    dd.io.save(path, obj.data, compression = None)

    return obj.update(data = path)


def dump_all_data(obj: NefBaseClass) -> NefBaseClass:
    out_dct = {}
    for key, val in obj.items():
        if key == 'data':
            if isinstance(val, str):
                continue
            path = RESOURCE_DIR + get_hash_of_timestamp() + '.hdf5'
            dd.io.save(path, val, compression = None)
            out_dct.update({key: path})
        elif isinstance(val, NefBaseClass):
            out_dct.update({key: dump_all_data(val)})
    return obj.update(**out_dct)


def load_data(obj: NefBaseClass) -> NefBaseClass:
    return obj.update(data = dd.io.load(obj.data))


def load_all_data(obj: NefBaseClass) -> NefBaseClass:
    out_dct = {}
    for key, type_ in obj.__class__.__annotations__.items():

        if key == 'data':
            if not isinstance(getattr(obj, 'data'), str):
                continue
            out_dct.update({'data': dd.io.load(getattr(obj, 'data'))})
        elif issubclass(type_, NefBaseClass):
            if getattr(obj, key, None) is not None:
                out_dct.update({key: load_all_data(getattr(obj, key))})
    return obj.update(**out_dct)


class DumpDataMixin(NefBaseClass):
    def dump_data(self) -> NefBaseClass:
        return dump_data(self)

    def dump_all_data(self) -> NefBaseClass:
        return dump_all_data(self)


class LoadDataMixin(NefBaseClass):
    def load_data(self) -> NefBaseClass:
        return load_data(self)

    def load_all_data(self) -> NefBaseClass:
        return load_all_data(self)
