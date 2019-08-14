# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: projection.py
@date: 5/20/2019
@desc:
'''
from nefct import nef_class, List
from nefct.geometry.scanner_config import ScannerConfig
import attr
import numpy as np

__all__ = (
'Projection', 'Projection2D', 'Projection3D', 'ProjectionSequence2D', 'ProjectionSequence3D')


@nef_class
class Projection:
    data: object
    scanner: ScannerConfig
    angle: float = attr.ib(default = 0.0)
    offset: list = attr.ib(default = 0.0)

    @property
    def source_position(self):
        pass

    @property
    def detector_position(self):
        pass


@nef_class
class Projection2D(Projection):
    data: object
    scanner: ScannerConfig
    angle: float = attr.ib(default = 0.0)
    offset: List(float, 2) = attr.ib(default = [0.0, 0.0])

    @property
    def source_position(self):
        x1, y1 = -self.scanner.SAD * np.cos(
            self.angle), -self.scanner.SAD * np.sin(self.angle)
        return x1 + self.offset[0], y1 + self.offset[1]

    @property
    def detector_position(self):
        x0, y0 = self.scanner.position
        x1, y1 = x0 + self.offset[0], y0 + self.offset[1]
        x2 = x1 * np.cos(self.angle) - y1 * np.sin(self.angle)
        y2 = x1 * np.sin(self.angle) + y1 * np.cos(self.angle)
        return x2 + self.offset[0], y2 + self.offset[1]


@nef_class
class Projection3D(Projection):
    data: object
    scanner: ScannerConfig
    angle: float = attr.ib(default = 0.0)
    offset: List(float, 3) = attr.ib(default = [0.0, 0.0, 0.0])

    @property
    def source_position(self):
        x1, y1 = -self.scanner.SAD * np.cos(
            self.angle), -self.scanner.SAD * np.sin(self.angle)
        return x1 + self.offset[0], y1 + self.offset[1], self.offset[2]

    @property
    def detector_position(self):
        x0, y0, z0 = self.scanner.position
        x1, y1, z1 = x0 + self.offset[0], y0 + self.offset[
            1], z0 + self.offset[2]
        x2 = x1 * np.cos(self.angle) - y1 * np.sin(self.angle)
        y2 = x1 * np.sin(self.angle) + y1 * np.cos(self.angle)
        return x2 + self.offset[0], y2 + self.offset[1], z1 + self.offset[2]


@nef_class
class ProjectionSequence2D:
    data: list
    scanner: ScannerConfig
    angle: list
    offset: list
    timestamps: list

    def __post_attr_init__(self):
        _n_view = self.data.size
        if self.angle is None:
            _angle = [ind * 2 * np.pi / _n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angle', _angle)
        if self.offset is None:
            object.__setattr__(self, 'offset', [0.0, 0.0])
        if not isinstance(self.offset[0], list):
            object.__setattr__(self, 'offset', [self.offset] * _n_view)

        if self.timestamps is None:
            _time_stamp = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'timestamps', _time_stamp)

    def source_position(self, ind: int):
        x0, y0, z0 = -self.scanner.SAD, 0.0, 0.0

        x1 = x0 * np.cos(self.angle[ind]) - y0 * np.sin(
            self.angle[ind]) + np.array([off[0] for off in self.offset])
        y1 = x0 * np.sin(self.angle[ind]) + y0 * np.cos(
            self.angle[ind]) + np.array([off[1] for off in self.offset])
        z1 = z0 + np.array([off[2] for off in self.offset])
        return x1, y1, z1

    def detector_position(self, ind: int):
        pass

    @property
    def n_view(self):
        return self.data.shape[3]

    def __getitem__(self, ind):
        return Projection3D(self.data[ind], self.scanner, self.angle[ind],
                            self.offset[ind])


@nef_class
class ProjectionSequence3D:
    data: list
    scanner: ScannerConfig
    angle: list
    offset: list
    timestamps: list

    def __post_attr_init__(self):
        _n_view = self.data.size
        if self.angle is None:
            _angle = [ind * 2 * np.pi / _n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angle', _angle)
        if self.offset is None:
            object.__setattr__(self, 'offset', [0.0, 0.0, 0.0])
        if not isinstance(self.offset[0], list):
            object.__setattr__(self, 'offset', [self.offset] * _n_view)

        if self.timestamps is None:
            _time_stamp = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'timestamps', _time_stamp)

    def source_position(self, ind: int):
        x0, y0, z0 = -self.scanner.SAD, 0.0, 0.0

        x1 = x0 * np.cos(self.angle[ind]) - y0 * np.sin(
            self.angle[ind]) + np.array([off[0] for off in self.offset])
        y1 = x0 * np.sin(self.angle[ind]) + y0 * np.cos(
            self.angle[ind]) + np.array([off[1] for off in self.offset])
        z1 = z0 + np.array([off[2] for off in self.offset])
        return x1, y1, z1

    def detector_position(self, ind: int):
        pass

    @property
    def n_view(self):
        return self.data.shape[3]

    def __getitem__(self, ind):
        return Projection3D(self.data[ind], self.scanner, self.angle[ind],
                            self.offset[ind])
