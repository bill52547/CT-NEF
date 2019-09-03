# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: sart.py
@date: 8/28/2019
@desc:
'''

from nefct import nef_class
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.project import Project
from nefct.functions.back_project import BackProject
import tensorflow as tf
from nefct.utils import tqdm


@nef_class
class SART:
    n_iter: int
    lambda_: float
    emap: Image
    project: Project
    back_project: BackProject

    def __call__(self, projection: ProjectionSequence) -> Image:
        x_tf = self.emap * 0
        for _ in tqdm(range(self.n_iter)):
            _projection_tf = self.project(x_tf)
            _bproj_tf = self.back_project(projection - _projection_tf)
            _bproj_tf2 = _bproj_tf.update(data = tf.div_no_nan(_bproj_tf.data,
                                                               self.emap.data))
            x_tf = x_tf.update(data = (x_tf + _bproj_tf2 * self.lambda_).data.numpy())

        return x_tf
