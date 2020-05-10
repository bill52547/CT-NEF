from nefct import nef_class
import numpy as np
from .detector_config import DetectorDirectAConfig, DetectorDirectBConfig

__all__ = ('ScannerConfig2D', 'ScannerConfig3D')


class ScannerConfig:
    pass


'''
Source(S) --> Axis(A) --> Detector Isocenter(I)
'''
@nef_class
class ScannerConfig2D(ScannerConfig):
    mode: str
    SID: float
    SAD: float
    detector: DetectorDirectAConfig

    @property
    def AID(self):
        return self.SID - self.SAD

    @property
    def positions(self):
        if self.mode == 'flat' or self.mode.startswith('f'):
            x1 = np.zeros(self.detector.number,
                          np.float32) + self.SID - self.SAD
            y1 = self.detector.meshgrid
            return x1, y1
        elif self.mode == 'cylin' or self.mode.startswith('c'):
            ang = self.detector.meshgrid
            x1 = self.SID * np.cos(ang) - self.SAD
            y1 = self.SID * np.sin(ang)
            return x1, y1
        elif '3d' in self.mode:
            raise ValueError('switch to ScannerConfig3D')
        else:
            raise NotImplementedError


@nef_class
class ScannerConfig3D(ScannerConfig):
    mode: str
    SID: float
    SAD: float
    detector_a: DetectorDirectAConfig
    detector_b: DetectorDirectBConfig

    @property
    def AID(self):
        return self.SID - self.SAD

    @property
    def positions(self):
        if self.mode == 'flat' or self.mode.startswith('f'):
            x1 = np.zeros((self.detector_a.number, self.detector_b.number), np.float32) + self.SID - \
                 self.SAD
            y1 = np.kron(self.detector_a.meshgrid,
                         [[1]] * self.detector_b.number).transpose()
            z1 = np.kron(self.detector_b.meshgrid,
                         [[1]] * self.detector_a.number)
            return x1, y1, z1
        elif self.mode == 'cylin' or self.mode.startswith('c'):
            ang = np.kron(self.detector_a.meshgrid,
                          [[1]] * self.detector_b.number).transpose()
            x1 = self.SID * np.cos(ang) - self.SAD
            y1 = self.SID * np.sin(ang)
            z1 = np.kron(self.detector_b.meshgrid,
                         [[1]] * self.detector_a.number)
            return x1, y1, z1
        elif '2d' in self.mode:
            raise ValueError('switch to ScannerConfig2D')
        else:
            raise NotImplementedError
