# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_magic_method_mixins.py
@date: 5/8/2019
@desc:
'''
from nefct.mixins.magic_method_mixins import *
import numpy as np
from nefct import nef_class


@nef_class
class SampleGetItemClass(GetItemMixin):
    data: np.ndarray


class TestGetItemMixin:
    def test_getitem(self):
        assert all(SampleGetItemClass(np.arange(10))[:5].data == np.arange(5))
