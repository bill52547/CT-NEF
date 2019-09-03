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
    _temp[z.data - lambda_ > 0] = z.data[z.data - lambda_ > 0] - lambda_
    return z.update(data = np.sign(z.data) * _temp)


@nef_class
class SoftShrink:
    lambda_: float

    def __call__(self, d: TotalVariation):
        return _soft_shrink(d, self.lambda_)
        # if isinstance(d, TotalVariation2D):
        #     sol = _soft_shrink(d, self.lambda_)
        #     return TotalVariation2D(sol)
        # elif isinstance(d, TotalVariation3D):
        #     sol = _soft_shrink(d.x, self.lambda_)
        #     return TotalVariation3D(sol)
        # elif isinstance(d, TotalVariation2DT):
        #     sol = _soft_shrink(d, self.lambda_)
        #     return TotalVariation2DT(sol)
        # elif isinstance(d, TotalVariation3DT):
        #     sol = _soft_shrink(d, self.lambda_)
        #     return TotalVariation3DT(sol)
        # else:
        #     raise NotImplementedError
