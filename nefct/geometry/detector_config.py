from nefct import nef_class
import numpy as np
import attr

__all__ = ('DetectorDirectAConfig', 'DetectorDirectBConfig', 'DetectorConfig')


@nef_class
class DetectorDirectAConfig:
    number: int
    size: float
    offset: float = attr.ib(default=0.0)

    @property
    def unit_size(self):
        return self.size / self.number

    @property
    def meshgrid(self):
        return (0.5 + np.arange(
            self.number)) * self.unit_size - self.size / 2 + self.offset


@nef_class
class DetectorDirectBConfig:
    number: int
    size: float
    offset: float = attr.ib(default=0.0)

    @property
    def unit_size(self):
        return self.size / self.number

    @property
    def meshgrid(self):
        return (0.5 + np.arange(
            self.number)) * self.unit_size - self.size / 2 + self.offset


@nef_class
class DetectorConfig:
    direct_a: DetectorDirectAConfig
    direct_b: DetectorDirectBConfig

    @property
    def shape(self):
        return [self.direct_a.number, self.direct_b.number]

    @property
    def size(self):
        return [self.direct_a.size, self.direct_b.size]

    @property
    def center(self):
        return [self.direct_a.offset, self.direct_b.offset]

    @property
    def unit_size(self):
        return [self.direct_a.unit_size, self.direct_b.unit_size]

    @property
    def meshgrid(self):
        return np.meshgrid(self.direct_a.meshgrid,
                           self.direct_b.meshgrid,
                           indexing='ij')
