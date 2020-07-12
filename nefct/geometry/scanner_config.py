from nefct import nef_class
import numpy as np
from .detector_config import DetectorDirectAConfig, DetectorDirectBConfig

__all__ = ('ScannerConfig',)


'''
Source(S) --> Axis(A) --> Detector Isocenter(I)
'''


@nef_class
class ScannerConfig:
    SID: float
    SAD: float
    detector_a: DetectorDirectAConfig
    detector_b: DetectorDirectBConfig

    @property
    def AID(self):
        return self.SID - self.SAD

    @property
    def positions(self):
        x1 = np.zeros((self.detector_a.number, self.detector_b.number), np.float32) + self.SID - \
            self.SAD
        y1 = np.kron(self.detector_a.meshgrid,
                     [[1]] * self.detector_b.number).transpose()
        z1 = np.kron(self.detector_b.meshgrid,
                     [[1]] * self.detector_a.number)
        return x1, y1, z1
