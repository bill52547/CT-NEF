# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_backproject_mixin.py
@date: 5/8/2019
@desc:
'''
from nefct.mixins.project_mixin import ProjectMixin
from nefct.toy_data import lors, image
import tensorflow as tf


class TestProjectMixin:
    def test_project_siddon_tf_eager(self):
        project = ProjectMixin()._project_siddon_tf
        if not tf.compat.v1.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        listmode_data = project(image, lors)
        assert listmode_data.shape[0] == len(lors)
        # assert np.max(listmode_data) == 0.0
