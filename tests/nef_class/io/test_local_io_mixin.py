# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_local_io_mixin.py
@date: 5/7/2019
@desc:
'''
import numpy as np
from nefct.nef_classes.io.local_io_mixin import SaveMixin, LoadMixin
from nefct.config import TMP_DIR
from nefct import nef_class


@nef_class
class ClassOne(SaveMixin, LoadMixin):
    data: np.ndarray
    val: int


@nef_class
class ClassTwo(SaveMixin, LoadMixin):
    val: int
    obj1: ClassOne


obj1 = ClassOne(np.zeros(5, ), 1)
obj2 = ClassTwo(5, obj1)


class Test_SaveMixin:
    def test_save(self, path = TMP_DIR + 'temp_obj1.hdf5'):
        obj1.save(path)

    def test_nested_save(self, path = TMP_DIR + 'temp_obj2.hdf5'):
        obj2.save(path)

    def test_nfs_save(self):
        obj1.save()


class Test_LoadMixin:
    def test_load(self, path = TMP_DIR + 'temp_obj1.hdf5'):
        obj1_bak = ClassOne.load(path)
        assert isinstance(obj1_bak, ClassOne)
        assert all(obj1_bak.data == obj1.data)
        assert obj1_bak.val == obj1.val

    def test_nest_load(self, path = TMP_DIR + 'temp_obj2.hdf5'):
        obj2_bak = ClassTwo.load(path)
        assert isinstance(obj2_bak, ClassTwo)
        assert obj2_bak.val == obj2.val
        assert obj2.obj1.val == obj1.val
        assert all(obj2.obj1.data == obj1.data)
