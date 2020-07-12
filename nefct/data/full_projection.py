# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: full_projection.py
@date: 5/20/2019
@desc:
'''
from nefct import nef_class, List
from nefct.geometry.scanner_config import ScannerConfig
import numpy as np


@nef_class
class FullProjection:
    data: object
    scanner: ScannerConfig
    angle: List(float)
    offset: List(float, 3)
    timestamps: List(float)
    volumes: List(float)

    def __post_attr_init__(self):
        _n_view = self.data.shape[3]
        if self.angle is None:
            _angle = [ind * 2 * np.pi / self.n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angles', _angle)
        if self.offset is None:
            object.__setattr__(self, 'offsets', [0.0, 0.0, 0.0])
        if self.timestamps is None:
            _time_stamp = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'timestamps', _time_stamp)

        if self.volumes is None:
            _volumes = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'volumes', _volumes)

    def pos_source(self, ind: int):
        x0, y0, z0 = -self.scanner.SAD, 0.0, 0.0
        x1, y1, z1 = x0 + self.offset[0], y0 + self.offset[1], z0 + self.offset[2]
        x2 = x1 * np.cos(self.angle[ind]) - y1 * np.sin(self.angle[ind])
        y2 = x1 * np.sin(self.angle[ind]) + y1 * np.cos(self.angle[ind])
        return x2, y2, z1

    def position(self, ind: int):
        x0, y0, z0 = self.scanner.positions
        x1, y1, z1 = x0 + self.offset[0], y0 + self.offset[1], z0 + self.offset[2]
        x2 = x1 * np.cos(self.angle[ind]) - y1 * np.sin(self.angle[ind])
        y2 = x1 * np.sin(self.angle[ind]) + y1 * np.cos(self.angle[ind])
        return x2, y2, z1

    @property
    def n_view(self):
        return self.data.shape[3]

    @property
    def flows(self):
        return [0.0] + [y - x for x, y in zip(self.volumes[1:], self.volumes[:-1])]
