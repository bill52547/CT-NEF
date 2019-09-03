# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_base.py
@date: 4/24/2019
@desc:
'''

import pytest
from nefct import nef_class, List, eq_


@nef_class
class SampleBaseClass:
    val1: int
    val2: float
    val3: List(int, 3)

    def __call__(self, a: int) -> int:
        return self.val1 + a


sample1 = SampleBaseClass(1, 2.0, [1, 2, 3])


class Test_BaseClass:
    # def test_new(self):
     #         wrong_sample1 = SampleBaseClass(1, 2.0, 3)
    #         wrong_sample2 = SampleBaseClass(1, 2.0, [1.0, 2.0, 3.0, 4.0])
    #         wrong_sample3 = SampleBaseClass(1, 2.0, val3 = [1.0, 2.0, 3.0, 4.0])

    def test_keys(self):
        assert tuple(SampleBaseClass.keys()) == ('val1', 'val2', 'val3')
        assert tuple(sample1.keys()) == ('val1', 'val2', 'val3')

    def test_values(self):
        assert sample1.values() == [1, 2.0, [1, 2, 3]]

    def test_items(self):
        assert sample1.items() == [('val1', 1), ('val2', 2.0), ('val3', [1, 2, 3])]

    def test_types(self):
        for type1, type2 in zip(SampleBaseClass.types(), [int, float, List(int, 3)]):
            assert eq_(type1, type2)
        for type1, type2 in zip(sample1.types(), [int, float, List(int, 3)]):
            assert eq_(type1, type2)

    def test_update(self):
        annother_sample1 = sample1.update()
        wrong_sample1 = sample1.update(val1 = 2)

        assert sample1 == annother_sample1
        assert not sample1 == wrong_sample1

    def test_asdict(self):
        dct1 = {'val1': 1, 'val2': 2.0, 'val3': [1, 2, 3]}
        assert dct1 == sample1.asdict()

    def test_from_dict(self):
        another_dict = {'val1': 2}
        assert SampleBaseClass.from_dict(another_dict).val1 == 2

    def test_astype(self):
        @nef_class
        class SampleBaseClass2:
            val1: int
            val2: float
            val3: List(int, 3)

        sample2 = sample1.astype(SampleBaseClass2)
        assert isinstance(sample2, SampleBaseClass2)
        assert sample2.val1 == sample1.val1

    def test_version(self):
        isinstance(sample1.__version__, str)

    def test_attrs_post_init__(self):
        sample1.__attrs_post_init__()

    def test_attr_eq(self):
        assert sample1.attr_eq(sample1)
        assert not sample1.update(val1 = 2).attr_eq(sample1)

    def test_call_annotations(self):
        assert sample1.__call_annotations__() == {'a': 'int', 'return': 'int'}

#
# @nef_classes(mode = 'config')
# class SampleConfigClass:
#     val1: int
#     val2: float
#     val3: List(int, 3)
#
#
# config_sample1 = SampleConfigClass(1, 2.0, [1, 2, 3])
#
#
# class Test_ConfigClass:
#     def test_new(self):
#         with pytest.raises(TypeError):
#             wrong_config_sample1 = SampleBaseClass(1, 2.0, 3)
#             wrong_config_sample2 = SampleBaseClass(1, 2.0, [1.0, 2.0, 3.0, 4.0])
#             wrong_config_sample3 = SampleBaseClass(1, 2.0, val3 = [1.0, 2.0, 3.0, 4.0])
#
#     def test_data_field_keyerror(self):
#         with pytest.raises(KeyError):
#             @nef_classes(mode = 'config')
#             class BadSampleConfigClass:
#                 data: int
#
#     def test_func_field_typeerror(self):
#         with pytest.raises(TypeError):
#             @nef_classes(mode = 'config')
#             class BadSampleConfigClass2:
#                 val1: int
#
#                 def bad_func(self):
#                     pass
#
#     def test_property_waiver(self):
#         @nef_classes(mode = 'config')
#         class BadSampleConfigClass3:
#             val1: int
#
#             @property
#             def bad_func(self):
#                 return 1
#
#     def test_keys(self):
#         assert tuple(SampleBaseClass.keys()) == ('val1', 'val2', 'val3')
#         assert tuple(config_sample1.keys()) == ('val1', 'val2', 'val3')
#
#     def test_types(self):
#         for type1, type2 in zip(SampleBaseClass.types(), [int, float, List(int, 3)]):
#             assert eq_(type1, type2)
#         for type1, type2 in zip(config_sample1.types(), [int, float, List(int, 3)]):
#             assert eq_(type1, type2)
#
#     def test_update(self):
#         annother_config_sample1 = config_sample1.update()
#         wrong_config_sample1 = config_sample1.update(val1 = 2)
#
#         assert config_sample1 == annother_config_sample1
#         assert not config_sample1 == wrong_config_sample1
#
#     def test_asdict(self):
#         dct1 = {'val1': 1, 'val2': 2.0, 'val3': [1, 2, 3]}
#         assert dct1 == config_sample1.asdict()
#
#     def test_values(self):
#         assert config_sample1.values() == [1, 2.0, [1, 2, 3]]
#
#     def test_items(self):
#         assert config_sample1.items() == [('val1', 1), ('val2', 2.0), ('val3', [1, 2, 3])]
#
#     def test_attrs_post_init__(self):
#         config_sample1.__attrs_post_init__()
#
#
# @nef_classes(mode = 'mixin')
# class SampleMixinClass:
#     def func1(self, val1: int) -> int:
#         return val1
#
#
# mixin_sample1 = SampleMixinClass()
#
#
# class Test_MixinClass:
#     def test_attr_field_attributeerror(self):
#         with pytest.raises(AttributeError):
#             @nef_classes(mode = 'mixin')
#             class BadSampleMixinClass:
#                 val1: int
#
#     def test_func_with_nonbasic_attributeerror(self):
#         with pytest.raises(TypeError):
#             @nef_classes(mode = 'mixin')
#             class BadSampleMixinClass:
#                 def func1(self, val1: SampleBaseClass) -> None:
#                     pass
#
#             @nef_classes(mode = 'mixin')
#             class BadSampleMixinClass2:
#                 def func1(self, val1: int) -> SampleBaseClass:
#                     return sample1
#
#     def test_func_keys(self):
#         assert mixin_sample1.func_keys() == ['func1']
#         assert SampleMixinClass.func_keys() == ['func1']
#
#     def test_func_types(self):
#         assert mixin_sample1.func_types() == [(int, int)]
#         assert SampleMixinClass.func_types() == [(int, int)]
#
#     def test_func_signatures(self):
#         assert mixin_sample1.func_signatures() == {'func1': {'val1': int, 'return': int}}
#         assert SampleMixinClass.func_signatures() == {'func1': {'val1': int, 'return': int}}
#
#
# @nef_classes(mode = 'data')
# class SampleDataClass:
#     data: int
#     val: int
#
#
# data_sample1 = SampleDataClass(1, 1)
#
#
# class Test_DataClass:
#     def test_new(self):
#         with pytest.raises(TypeError):
#             wrong_data_sample1 = SampleBaseClass(1, 2.0)
#             wrong_data_sample2 = SampleBaseClass(1.0)
#         right_data_sample1 = SampleDataClass(1)
#
#     def test_data_field_non_existing_keyerror(self):
#         with pytest.raises(KeyError):
#             @nef_classes(mode = 'data')
#             class BadSampleDataClass:
#                 val1: int
#
#     def test_func_field_typeerror(self):
#         with pytest.raises(TypeError):
#             @nef_classes(mode = 'data')
#             class BadSampleDataClass2:
#                 data: int
#                 val1: int
#
#                 def bad_func(self):
#                     pass
#
#     def test_property_waiver(self):
#         @nef_classes(mode = 'data')
#         class BadSampleDataClass3:
#             data: int
#             val1: int
#
#             @property
#             def bad_func(self):
#                 return 1
#
#     def test_add(self):
#         data_sample2 = data_sample1 + 1
#         assert data_sample2.data == 2
#
#
# @nef_classes(mode = 'func')
# class SampleFuncClass:
#     val1: int
#
#     def __call__(self, *args, **kwargs):
#         pass
#
#
# func_sample1 = SampleFuncClass(1)
#
#
# @nef_classes(mode = 'func')
# class BadFuncClass:
#     val1: int
#
#
# class Test_FuncClass:
#     def test_data_field_keyerror(self):
#         with pytest.raises(KeyError):
#             @nef_classes(mode = 'config')
#             class BadFuncClass2:
#                 data: int
