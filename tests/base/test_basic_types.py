# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_basic_types.py
@date: 4/24/2019
@desc:
'''
import pytest
from nefct import Optional, NoneType, Any, isinstance_, List, Tuple


class Test_None_Type:
    def test_none_type(self):
        assert NoneType is type(None)


class Test_Any:
    def test_object(self):
        assert Any == object


basic_type_list = (int, float, bool, str, list, tuple)

basic_type_samples = {int: (1, 0, 5),
                      float: (1.0, 2.0, 0.0, -1.5),
                      bool: (True, False),
                      str: ('hello', '', '123'),
                      list: ([1, 2, 3], [True, False], [-1.0, 2.0]),
                      tuple: ((1, 2.0, '123'), (list, int), (True, 1))}


class Test_Optional:
    def test_basic_types(self):
        for type_ in basic_type_list:
            assert Optional(type_) == (type_, NoneType)
            assert type_ in Optional(type_)
            assert NoneType in Optional(type_)
            assert isinstance(None, Optional(type_))

            for any_type in basic_type_list:
                if issubclass(any_type, type_):
                    for sample in basic_type_samples[any_type]:
                        assert isinstance(sample, Optional(any_type))
                else:
                    for sample in basic_type_samples[any_type]:
                        assert not isinstance(sample, Optional(type_))


class Test_List:
    def test_create(self):
        list_int_3 = List(int, 3)
        assert isinstance(list_int_3, type)
        assert issubclass(list_int_3, list)

        for any_type in basic_type_list:
            if issubclass(any_type, list):
                for sample in basic_type_samples[any_type]:
                    if len(sample) == 3 and all(map(lambda x: isinstance(x, int), sample)):
                        pass
                    else:
                        with pytest.raises(TypeError):
                            list_int_3(sample)
            else:
                for sample in basic_type_samples[any_type]:
                    with pytest.raises(TypeError):
                        list_int_3(sample)

        assert list_int_3.length == 3
        assert list_int_3.dtype == int

    def test_isinstance(self):
        list_int_3 = List(int, 3)
        for any_type in basic_type_list:
            if issubclass(any_type, list):
                for sample in basic_type_samples[any_type]:
                    if len(sample) == 3 and all(map(lambda x: isinstance(x, int), sample)):
                        assert list_int_3.isinstance(sample)
                    else:
                        assert not list_int_3.isinstance(sample)
            else:
                for sample in basic_type_samples[any_type]:
                    assert not list_int_3.isinstance(sample)


class Test_Tuple:
    def test_create(self):
        tuple_sample = Tuple((int, float, str))
        assert isinstance(tuple_sample, type)
        assert issubclass(tuple_sample, tuple)

        for any_type in basic_type_list:
            if issubclass(any_type, tuple):
                for sample in basic_type_samples[any_type]:
                    if sample == (1, 2.0, '123'):
                        pass
                    else:
                        with pytest.raises(TypeError):
                            tuple_sample(sample)
            else:
                for sample in basic_type_samples[any_type]:
                    with pytest.raises(TypeError):
                        tuple_sample(sample)
        assert tuple_sample.dtype == (int, float, str)

    def test_isinstance(self):
        tuple_sample = Tuple((int, float, str))
        for any_type in basic_type_list:
            if issubclass(any_type, tuple):
                for sample in basic_type_samples[any_type]:
                    if sample == (1, 2.0, '123'):
                        assert tuple_sample.isinstance(sample)
                    else:
                        assert not tuple_sample.isinstance(sample)
            else:
                for sample in basic_type_samples[any_type]:
                    assert not tuple_sample.isinstance(sample)


class Test_isinstance_:
    def test_basic_type_isinstance_(self):
        for type_ in basic_type_list:
            for any_type in basic_type_list:
                if issubclass(any_type, type_):
                    for sample in basic_type_samples[any_type]:
                        assert isinstance_(sample, Optional(any_type))
                else:
                    for sample in basic_type_samples[any_type]:
                        assert not isinstance_(sample, Optional(type_))

    def test_generic_type_isinstance_(self):
        list_int_3 = List(int, 3)
        tuple_sample = Tuple((int, float, str))
        for type_ in basic_type_list:
            for sample in basic_type_samples[type_]:
                if list_int_3.isinstance(sample):
                    assert isinstance_(sample, list_int_3)
                else:
                    assert not isinstance_(sample, list_int_3)
                if tuple_sample.isinstance(sample):
                    assert isinstance_(sample, tuple_sample)
                else:
                    assert not isinstance_(sample, tuple_sample)
