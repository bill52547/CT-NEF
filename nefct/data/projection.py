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
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import attr
import numpy as np
from nefct.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from nefct.ops.common.imshow_mixin import ImshowMixin
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin

__all__ = ('ProjectionSequence2D', 'ProjectionSequence3D')


@nef_class
class ProjectionSequence(ArithmeticMixin):

    @property
    def source_position(self):
        pass

    @property
    def detector_position(self):
        pass


@nef_class
class ProjectionSequence2D(ProjectionSequence):
    data: np.ndarray
    scanner: ScannerConfig2D
    angles: list
    timestamps: list

    def __attrs_post_init__(self):
        _n_view = self.data.shape[-1]
        if self.angles is None:
            _angle = [ind * 2 * np.pi / _n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angles', _angle)

    def source_position(self, ind: int):
        x0, y0, z0 = -self.scanner.SAD, 0.0, 0.0

        x1 = x0 * np.cos(self.angles[ind]) - y0 * np.sin(
            self.angles[ind])
        y1 = x0 * np.sin(self.angles[ind]) + y0 * np.cos(
            self.angles[ind])
        z1 = z0
        return x1, y1, z1

    def detector_position(self, ind: int):
        pass

    @property
    def n_view(self):
        return self.data.shape[3]

    def __getitem__(self, item):
        return ProjectionSequence2D(self.data[:, item], self.scanner, self.angles[item],
                                    self.timestamps[item])


@nef_class
class ProjectionSequence3D(CentralSlicesPropertyMixin, ImshowMixin,
                           ShapePropertyMixin, ProjectionSequence):
    data: np.ndarray
    scanner: ScannerConfig3D
    angles: list
    offsets: list
    timestamps: list

    def __attrs_post_init__(self):
        _n_view = self.data.shape[-1]
        if self.angles is None:
            _angle = [ind * 2 * np.pi / _n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angles', _angle)
        if self.offsets is None:
            object.__setattr__(self, 'offsets', [0.0])
        if not isinstance(self.offsets, (list, np.ndarray)):
            object.__setattr__(self, 'offsets', [self.offsets] * _n_view)

        if self.timestamps is None:
            object.__setattr__(self, 'timestamps', 0.0)
        if not isinstance(self.timestamps, (list, np.ndarray)):
            object.__setattr__(self, 'timestamps',
                               self.timestamps * np.arange(_n_view))

    def source_position(self, ind: int):
        x0, y0, z0 = -self.scanner.SAD, 0.0, 0.0

        x1 = x0 * np.cos(self.angles[ind]) - y0 * np.sin(
            self.angles[ind]) + np.array([off[0] for off in self.offsets])
        y1 = x0 * np.sin(self.angles[ind]) + y0 * np.cos(
            self.angles[ind]) + np.array([off[1] for off in self.offsets])
        z1 = z0 + np.array([off[2] for off in self.offsets])
        return x1, y1, z1

    def detector_position(self, ind: int):
        pass

    @property
    def n_view(self):
        return self.data.shape[2]

    def __getitem__(self, item):
        v_data = np.array([vf[0] for vf in self.timestamps]).astype(np.float32)
        f_data = np.array([vf[1] for vf in self.timestamps]).astype(np.float32)
        new_timestamps = list(zip(v_data[item], f_data[item]))
        return ProjectionSequence3D(self.data[:, :, item], self.scanner, self.angles[item], self.offsets[item], new_timestamps)
