# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_arithmetic_mixin.py
@date: 5/8/2019
@desc:
'''
import pytest
from nefct.mixins.arithmetic_mixins import ArithmeticMixin
from nefct import nef_class


@nef_class
class SampleClass(ArithmeticMixin):
    data: float


sample1 = SampleClass(-2.)
sample2 = SampleClass(4.)


class TestArithemeticMixin:
    def test_without_data_field_error(self):
        @nef_class
        class SampleClass2(ArithmeticMixin):
            val1: int
        #
        # with pytest.raises(TypeError):
        #     temp = SampleClass2(1.) + 1.

    def test_neg(self):
        assert (-sample1).data == 2.

    def test_abs(self):
        assert sample1.abs().data == 2.

    def test_add(self):
        assert (sample1 + 1.).data == -1.
        assert (sample1 + sample2).data == 2.

    def test_sub(self):
        assert (sample1 - 1.).data == -3.
        assert (sample1 - sample2).data == -6.

    def test_mul(self):
        assert (sample1 * 1.).data == -2.
        assert (sample1 * sample2).data == -8.

    def test_truediv(self):
        assert (sample1 / 2.).data == -1.
        assert (sample2 / sample1).data == -2.

    def test_floordiv(self):
        assert (sample1 // 4).data == -1.
        assert (sample1 // sample2).data == -1.

    def test_mod(self):
        assert (sample1 % 4).data == 2.
        assert (sample1 % sample2).data == 2.

    def test_pow(self):
        assert (sample1 ** 4).data == 16.
        assert (sample1 ** sample2).data == 16.
