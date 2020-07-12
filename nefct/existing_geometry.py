import numpy as np
from nefct.geometry.scanner_config import ScannerConfig
from nefct.geometry.detector_config import DetectorDirectAConfig, DetectorDirectBConfig
import nefct as nef
from nefct import load

__all__ = ('varian_3d_scanner', 'varian_angles',
           'elekta_3d_scanner', 'elekta_angles')

#
# __all__ = (
#     'cbct_a_detector', 'cbct_b_detector', 'cbct_2d_scanner', 'cbct_3d_scanner', 'cbct_angles',
#     'cbct_2d_full_fan_scanner', 'cbct_3d_full_fan_scanner', 'xcat_volumes', 'xcat_flows')
# cbct_a_detector = DetectorDirectAConfig(512, 0.388 * 1024, 147.44)
# cbct_b_detector = DetectorDirectBConfig(512, 0.388 * 1024, 0)
# cbct_2d_scanner = ScannerConfig2D('flat', 1500, 1000, cbct_a_detector)
# cbct_3d_scanner = ScannerConfig3D('flat', 1500, 1000, cbct_a_detector, cbct_b_detector)
# cbct_2d_full_fan_scanner = ScannerConfig2D('flat', 1500, 1000, cbct_a_detector.update(offset = 0))
# cbct_3d_full_fan_scanner = ScannerConfig3D('flat', 1500, 1000, cbct_a_detector.update(offset = 0),
#                                            cbct_b_detector)

elekta_a_detector = DetectorDirectAConfig(512, 397.312, 0)
elekta_b_detector = DetectorDirectBConfig(512, 397.312, 0)
elekta_angles = np.linspace(-3.141, 0.3469, 1327)
elekta_3d_scanner = ScannerConfig(
    1536, 1000, elekta_a_detector, elekta_b_detector)

varian_a_detector = DetectorDirectAConfig(512, 409.6, 147.2)
varian_b_detector = DetectorDirectBConfig(512, 409.6, 0)
varian_angles = np.linspace(0, 2 * 3.14159, 678)
varian_3d_scanner = ScannerConfig(
    1500, 1000, varian_a_detector, varian_b_detector)

nview = 2304
nrot = 6
# siemens_a_detector = DetectorDirectAConfig(736, 0.8722, 0)
