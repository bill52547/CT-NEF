# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_backproject_mixin.py
@date: 5/8/2019
@desc:
'''
import pytest
from nefct.mixins.backproject_mixin import BackProjectMixin
import tensorflow as tf
from nefct.toy_data import listmode, image
import numpy as np


class TestBackProjectMixin:
    def test_back_project_siddon_tf(self):
        bproject = BackProjectMixin()._back_project_siddon_tf
        if not tf.compat.v1.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        image_data = bproject(listmode, image)

        assert np.max(image_data) == 0.0
        assert image_data.shape == image.shape
