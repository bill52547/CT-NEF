# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: single_projection.py
@date: 5/20/2019
@desc:
'''
from nefct import nef_class, List
from nefct.geometry.scanner_config import ScannerConfig
import attr
import numpy as np


@nef_class
class SingleProjection:
    data: object
    scanner: ScannerConfig
    angle: float = attr.ib(default=0.0)
    offset: List(float, 3) = attr.ib(default=[0.0, 0.0, 0.0])

    @property
    def position(self):
        x0, y0, z0 = self.scanner.positions
        x1, y1, z1 = x0 + self.offset[0], y0 + self.offset[
            1], z0 + self.offset[2]
        x2 = x1 * np.cos(self.angle) - y1 * np.sin(self.angle)
        y2 = x1 * np.sin(self.angle) + y1 * np.cos(self.angle)
        return x2, y2, z1
