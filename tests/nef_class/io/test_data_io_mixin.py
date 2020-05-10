# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_data_io_mixin.py
@date: 5/7/2019
@desc:
'''
import numpy as np
from nefct import nef_class
from nefct.nef_classes.io.data_io_mixin import DumpDataMixin, LoadDataMixin


@nef_class
class ClassOne(DumpDataMixin, LoadDataMixin):
    data: np.ndarray
    val: int


@nef_class
class ClassTwo(DumpDataMixin, LoadDataMixin):
    data: np.ndarray
    val: int
    obj1: ClassOne


obj1 = ClassOne(np.zeros(5, ), 1)
obj1_ = obj1.dump_data()
obj2 = ClassTwo(np.ones(5, ), 1, obj1)
obj2_ = obj2.dump_data()


class Test_SaveMixin:
    def test_data_save(self):
        assert isinstance(obj1_.data, str)

    def test_nested_data_save(self):
        assert isinstance(obj2_.data, str)
        assert not isinstance(obj2_.obj1.data, str)

    def test_nfs_data_save(self):
        obj2.dump_data()


class Test_LoadMixin:
    def test_data_load(self):
        obj1_bak = obj1_.load_data()
        assert isinstance(obj1_bak, ClassOne)
        assert all(obj1_bak.data == obj1.data)
        assert obj1_bak.val == obj1.val

        obj2_bak = obj2_.load_data()
        assert isinstance(obj2_bak, ClassTwo)
        assert all(obj2_bak.data == obj2.data)
        assert obj2_bak.val == obj2.val
