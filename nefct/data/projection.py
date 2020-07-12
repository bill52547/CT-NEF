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
from nefct.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from nefct.ops.common.imshow_mixin import ImshowMixin
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin

__all__ = ('ProjectionSequence',)


@nef_class
class ProjectionSequence(CentralSlicesPropertyMixin, ImshowMixin,
                         ShapePropertyMixin, ArithmeticMixin):
    data: np.ndarray
    scanner: ScannerConfig
    angles: np.ndarray
    offsets_a: np.ndarray
    offsets_b: np.ndarray
    timestamps: np.ndarray

    def __attrs_post_init__(self):
        _n_view = self.data.shape[-1]
        if self.angles is None:
            _angle = [ind * 2 * np.pi / _n_view for ind in range(_n_view)]
            object.__setattr__(self, 'angles', _angle)
        if self.offsets_a is None:
            object.__setattr__(self, 'offsets_a', np.zeros(
                _n_view, dtype=np.float32))
        if self.offsets_b is None:
            object.__setattr__(self, 'offsets_b', np.zeros(
                _n_view, dtype=np.float32))
        if self.timestamps is None:
            object.__setattr__(self, 'timestamps', 0.0)
        if not isinstance(self.timestamps, (list, np.ndarray)):
            object.__setattr__(self, 'timestamps',
                               self.timestamps * np.arange(_n_view))

    @property
    def n_view(self):
        return self.data.shape[2]

    def __getitem__(self, item):
        new_timestamps = self.timestamps[item] if self.timestamps.ndim == 1 else self.timestamps[item, :]
        return ProjectionSequence(self.data[:, :, item],
                                  self.scanner, self.angles[item], self.offsets_a[item], self.offsets_b[item], new_timestamps)
