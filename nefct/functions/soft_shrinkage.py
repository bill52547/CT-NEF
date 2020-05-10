# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: soft_shrinkage.py
@date: 9/1/2019
@desc:
'''
from nefct import nef_class
from nefct import Image
from nefct.data.total_variation import TotalVariation, TotalVariation2D, TotalVariation3D, \
    TotalVariation2DT, TotalVariation3DT
import numpy as np

'''soft shrinkage to solve |x|_1 + norm(x - d, 2) * lambda_ / 2 '''


def _soft_shrink(z: Image, lambda_: float):
    _temp = z.data * 0
    _temp[np.abs(z.data) > lambda_] = z.data[np.abs(z.data) > lambda_] - lambda_
    return z.update(data = np.sign(z.data) * _temp)

@nef_class
class SoftShrink:
    lambda_: float

    def __call__(self, d: TotalVariation):
        return _soft_shrink(d, self.lambda_)