import numpy as np
from nefct.geometry.scanner_config import ScannerConfig2D, ScannerConfig3D
from nefct.geometry.detector_config import DetectorDirectAConfig, DetectorDirectBConfig
import nefct as nef
from nefct import load

__all__ = ('varian_2d_scanner', 'varian_3d_scanner', 'varian_angles',
           'elekta_2d_scanner', 'elekta_3d_scanner', 'elekta_angles',
           'siemens_2d_scanner', 'siemens_3d_scanner', 'siemens_angles', 'siemens_offsets',
           'xcat_image')

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
elekta_3d_scanner = ScannerConfig3D('flat', 1536, 1000, elekta_a_detector, elekta_b_detector)
elekta_angles = np.linspace(-3.141, 0.3469, 1327)

varian_a_detector = DetectorDirectAConfig(512, 409.6, 147.44)
varian_b_detector = DetectorDirectBConfig(512, 409.6, 0)
varian_2d_scanner = ScannerConfig2D('flat', 1500, 1000, varian_a_detector)
varian_3d_scanner = ScannerConfig3D('flat', 1500, 1000, varian_a_detector, varian_b_detector)
varian_angles = np.linspace(0, 2 * 3.14159, 678)

siemens_a_detector = DetectorDirectAConfig(736, 0.8722, 0)
siemens_b_detector = DetectorDirectBConfig(64, 38.4, 0)
siemens_2d_scanner = ScannerConfig2D('cyli', 1085.6, 595, siemens_a_detector)
siemens_3d_scanner = ScannerConfig3D('cyli', 1085.6, 595, siemens_a_detector, siemens_b_detector)
siemens_angles = np.linspace(0, 3.14159 * 16, 2304 * 8)
siemens_offsets = 38.4 * 8 * np.arange(8 * 2304) / 8 / 2304
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

#
# cbct_angles = np.linspace(0, np.pi * 2, 599).astype(np.float32)
# xcat_volumes = np.array([4.1, 4.9, 5.5, 5.7, 5.7, 5.1, 4.2, 3.4, 2.9, 2.4, 1.9,
#                          1.5, 1., 0.5, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.1, 0.1,
#                          -0.1, -0.3, -0.3, -0.3, -0.2, -0.2, -0.1, -0., 0.1, 0.4, 0.6,
#                          1., 1.7, 2.5, 3.3, 4., 4.7, 5.4, 6.1, 6.3, 5.6, 4.4,
#                          3.3, 2.8, 2.3, 1.9, 1.4, 1.1, 0.8, 0.4, 0.2, 0.1, 0.2,
#                          0.2, 0.2, 0.2, 0.2, 0.2, -0., -0.2, -0.4, -0.3, -0.2, -0.2,
#                          0., 0.4, 1., 1.9, 2.6, 3.1, 3.8, 4.6, 5.5, 6.4, 7.2,
#                          7.2, 6.2, 4.9, 3.7, 2.7, 2.1, 1.8, 1.4, 1.1, 0.8, 0.6,
#                          0.4, 0.2, 0., -0.1, -0.1, 0., 0.1, 0.1, 0.1, 0.1, 0.,
#                          -0.1, -0.4, -0.4, -0.1, 0.2, 0.7, 1.4, 2.2, 3.1, 3.8, 4.4,
#                          5., 5.8, 6.8, 7.7, 7.7, 6.7, 5.3, 4.2, 3.4, 2.5, 1.8,
#                          1.3, 1.1, 1., 0.8, 0.6, 0.4, 0.3, 0.1, -0., -0., 0.1,
#                          0.1, 0.1, 0.2, 0.2, 0.2, 0., -0.2, -0.3, -0.1, 0.1, 0.3,
#                          0.7, 1.3, 2., 2.8, 3.5, 3.9, 4.4, 5.1, 6., 6.8, 6.9,
#                          6., 4.8, 3.9, 3., 2.3, 1.8, 1.5, 1.4, 1.1, 0.9, 0.7,
#                          0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.,
#                          -0.1, -0.2, -0.1, 0.2, 0.7, 1.4, 2.3, 3.1, 4., 4.8, 5.5,
#                          6.3, 7., 7.6, 7.2, 6.1, 4.8, 3.9, 3.2, 2.5, 1.7, 1.1,
#                          0.7, 0.6, 0.5, 0.4, 0.4, 0.3, 0.3, 0.1, -0.1, -0.1, 0.,
#                          0.1, 0.1, 0.1, 0., -0., -0.1, -0.2, -0.3, 0., 0.4, 1.,
#                          1.7, 2.5, 3.2, 3.8, 4.1, 4.3, 4.4, 4.4, 3.9, 3.2, 2.6,
#                          2.3, 2., 1.7, 1., 0.6, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3,
#                          0.2, 0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1, 0., 0.2, 0.3,
#                          0.5, 0.8, 1.2, 2., 2.9, 3.7, 4.3, 5., 5.7, 6.6, 7.1,
#                          6.8, 5.6, 4.4, 3.8, 3.2, 2.6, 2.1, 1.7, 1.3, 0.8, 0.5,
#                          0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.1, -0.1, -0.2, -0.1,
#                          -0., -0.1, -0.1, -0.1, -0., 0.3, 0.6, 1.1, 1.8, 2.6, 3.5,
#                          4.2, 5., 5.8, 6.6, 7., 6.5, 5.2, 4., 3.4, 2.9, 2.4,
#                          2., 1.6, 1.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
#                          0., -0., -0.2, -0.5, -0.5, -0.3, -0.1, 0.2, 0.8, 1.4, 2.1,
#                          2.9, 3.5, 3.9, 4.4, 5.1, 5.8, 6.1, 5.7, 4.7, 3.9, 3.1,
#                          2.2, 1.6, 1.2, 1., 0.8, 0.6, 0.5, 0.3, 0.3, 0.1, -0.1,
#                          -0.1, -0., -0., 0., -0., -0.1, -0.2, -0.1, 0., 0.4, 1.,
#                          1.9, 2.8, 3.6, 4.2, 4.8, 5.4, 5.8, 5.9, 5.7, 5., 4.1,
#                          3.5, 3.2, 2.8, 2.3, 1.8, 1.2, 0.7, 0.5, 0.5, 0.4, 0.3,
#                          0.3, 0.2, 0., -0.2, -0.3, -0.3, -0.2, -0.2, -0.2, -0.1, -0.,
#                          0.1, 0.2, 0.4, 0.8, 1.4, 2.2, 3.1, 3.8, 4.5, 5.1, 5.8,
#                          6.3, 6.3, 5.6, 4.4, 3.5, 2.8, 2.2, 1.7, 1.2, 0.8, 0.4,
#                          0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.4, -0.7,
#                          -0.6, -0.5, -0.4, -0.4, -0.3, -0.3, -0.1, 0., 0.2, 0.4, 0.8,
#                          1.5, 2.3, 3.1, 3.8, 4.4, 5.1, 5.8, 5.9, 5.4, 4.5, 3.8,
#                          3.2, 2.6, 2.1, 1.8, 1.4, 0.9, 0.5, 0.2, 0.2, 0.2, 0.2,
#                          0.2, 0.1, 0.1, 0., -0.2, -0.5, -0.4, -0.3, -0.2, -0.2, -0.2,
#                          -0.2, -0.2, -0.1, -0.1, 0., 0.3, 0.8, 1.6, 2.6, 3.5, 4.3,
#                          5.2, 5.9, 6.4, 6.6, 6.8, 6.6, 5.8, 4.5, 3.7, 3.1, 2.6,
#                          1.8, 1.1, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, -0., -0.1, -0.2,
#                          -0.5, -0.5, -0.3, -0.3, -0.3, -0.2, -0.3, -0.4, -0.4, -0.5, -0.4,
#                          0.1, 0.7, 1.6, 2.6, 3.6, 4.4, 5.4, 6.2, 7.1, 8., 9.,
#                          10., 10.7, 10.4, 9.2, 7.8, 6.7, 5.3, 4., 3., 2.4, 1.9,
#                          1.4, 1., 0.6, 0.3, 0.1, -0.3, -0.6, -0.6, -0.5, -0.5, -0.5,
#                          -0.5, -0.6, -0.6, -0.7, -1., -1., -0.9, -0.8, -0.8, -0.6, -0.3,
#                          0.1, 0.6, 1.1, 1.8, 2.7, 3.6, 4.4, 5.4, 6.6, 8., 9.3,
#                          10.5, 11.6, 11.8, 11.4, 10.8, 9.5, 7.6, 5.8, 4.6, 3.6, 2.7,
#                          1.7, 1.1, 0.7, 0.5, 0.3, 0.1, -0., -0.2, -0.4, -0.6, -1.,
#                          -1.1, -0.9, -0.8, -0.7, -0.7], dtype = np.float32)
# xcat_flows = np.hstack(([0], xcat_volumes[1:] - xcat_volumes[:-1]))
