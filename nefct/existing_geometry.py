import numpy as np
from nefct.geometry.scanner_config import ScannerConfig2D, ScannerConfig3D
from nefct.geometry.detector_config import DetectorDirectAConfig, DetectorDirectBConfig
import nefct as nef
from nefct import load

__all__ = ('varian_2d_scanner', 'varian_3d_scanner', 'varian_angles',
           'elekta_2d_scanner', 'elekta_3d_scanner', 'elekta_angles',
           'siemens_2d_scanner', 'siemens_3d_scanner', 'siemens_angles',
           'siemens_offsets', 'xcat_image')

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
elekta_2d_scanner = ScannerConfig2D('flat', 1536, 1000, elekta_a_detector)
elekta_3d_scanner = ScannerConfig3D('flat', 1536, 1000, elekta_a_detector,
                                    elekta_b_detector)
elekta_angles = np.linspace(-3.141, 0.3469, 1327)

varian_a_detector = DetectorDirectAConfig(512, 409.6, 147.2)
varian_b_detector = DetectorDirectBConfig(512, 409.6, 0)
varian_2d_scanner = ScannerConfig2D('flat', 1500, 1000, varian_a_detector)
varian_3d_scanner = ScannerConfig3D('flat', 1500, 1000, varian_a_detector,
                                    varian_b_detector)
varian_angles = np.linspace(0, 2 * 3.14159, 678)

nview = 2304 
nrot = 6
# siemens_a_detector = DetectorDirectAConfig(736, 0.8722, 0)
siemens_a_detector = DetectorDirectAConfig(736, 800, 0)
siemens_b_detector = DetectorDirectBConfig(64, 200, 0)
siemens_2d_scanner = ScannerConfig2D('cyli', 1085.6, 595, siemens_a_detector)
siemens_3d_scanner = ScannerConfig3D('flat', 1085.6, 595, siemens_a_detector,
                                     siemens_b_detector)
siemens_angles = np.linspace(0, 3.14159 * 2 * nrot, nview * nrot).astype(np.float32)
siemens_offsets = nrot * 38.4 / 3 * np.arange(nrot * nview).astype(np.float32) / nrot / nview
siemens_offsets -= np.mean(siemens_offsets)

__all__ += ('foldername_5d_data', 'foldername_test')
foldername_5d_data = '/home/bill52547/Workspace/thesis_programs/data/5d_data/'
foldername_test = '/home/bill52547/Workspace/thesis_programs/data/test_data/'


def xcat_image(name: str = None):
    if name == '2d':
        return load(nef.Image2D, '~/thesis_programs/xcat/xcat_2d.hdf5')
    elif name == '2dt':
        return load(nef.Image2DT, '~/thesis_programs/xcat/xcat_2dt.hdf5')
    elif name == '3d':
        return load(nef.Image3D, '~/thesis_programs/xcat/xcat_3d.hdf5')
    elif name == '3dt':
        return load(nef.Image3DT, '~/thesis_programs/xcat/xcat_3dt.hdf5')
    else:
        return load(nef.Image2D, '~/thesis_programs/xcat/xcat_2d.hdf5')
