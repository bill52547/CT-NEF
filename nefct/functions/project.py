from ctypes import *
import ctypes
import numpy as np
from nefct import nef_class, Any
from nefct.data.image import Image2D, Image3D, Image, Image2DT, Image3DT
from nefct.data.projection import ProjectionSequence2D, \
    ProjectionSequence3D, ProjectionSequence
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import numpy as np
import tensorflow as tf
from nefct.utils import tqdm
from typing import Callable
from nefct.data.deform_para import DeformParameter

CTYPE_SO_FILE = '/home/bill52547/Workspace/NefCT/nefct/ctypes/'


@nef_class
class Project:
    scanner: ScannerConfig
    angles: np.ndarray
    offsets_b: np.ndarray
    offsets_a: np.ndarray

    def __call__(self, image: Image3D) -> ProjectionSequence3D:
        dll = ctypes.CDLL(CTYPE_SO_FILE + 'forward_dd.so',
                          mode=ctypes.RTLD_GLOBAL)
        func = dll.project
        func.argtypes = [POINTER(c_float), POINTER(c_float),
                         POINTER(c_float), POINTER(c_float), POINTER(c_float),
                         c_int, c_int, c_int,
                         c_float, c_float, c_float,
                         c_float, c_float,
                         c_float, c_int,
                         c_float, c_int,
                         c_int]
        na = self.scanner.detector_a.number
        nb = self.scanner.detector_b.number
        nv = self.angles.size
        dx = image.unit_size[0]
        img_p = image.data.T.ravel().ctypes.data_as(POINTER(c_float))
        proj_data = np.zeros((na * nb * nv,), dtype=np.float32)
        proj_p = proj_data.ctypes.data_as(POINTER(c_float))
        angles_p = self.angles.astype(
            np.float32).ctypes.data_as(POINTER(c_float))
        off_a_p = (self.offsets_a).astype(
            np.float32).ctypes.data_as(POINTER(c_float))
        off_b_p = (self.offsets_b).astype(
            np.float32).ctypes.data_as(POINTER(c_float))

        func(proj_p, img_p,
             angles_p, off_a_p, off_b_p,
             image.shape[0], image.shape[1], image.shape[2],
             image.unit_size[0], image.unit_size[1], image.unit_size[2],
             self.scanner.SID, self.scanner.SAD,
             self.scanner.detector_a.unit_size, na,
             self.scanner.detector_b.unit_size, nb,
             nv)

        return ProjectionSequence3D(proj_data, self.scanner, self.angles, self.offsets_b)


# def project(listmode, image, tof_reso, index2pos):
#     dll = ctypes.CDLL(CTYPE_SO_FILE + 'forward_dd.so', mode=ctypes.RTLD_GLOBAL)
#     func = dll.backprojection_ctypes_wrapper
#     func.argtypes = [POINTER(c_float), POINTER(c_float),
#                      POINTER(c_ushort), POINTER(c_ushort), POINTER(c_float),
#                      POINTER(c_float), POINTER(c_float), POINTER(c_float),
#                      c_int, c_int, c_int, c_float, c_float, c_float,
#                      c_float, c_int, c_int, c_float, c_float]
#     img_out = image.data.ravel() * 0
#     img_p = img_out.ctypes.data_as(POINTER(c_float))
#     proj_p = listmode.data.astype(np.float32).ctypes.data_as(POINTER(c_float))
#     index1_p = listmode.lors.index1.astype(
#         np.uint16).ctypes.data_as(POINTER(c_ushort))
#     index2_p = listmode.lors.index2.astype(
#         np.uint16).ctypes.data_as(POINTER(c_ushort))
#     tof_val_p = listmode.lors.tof_vals.astype(
#         np.float32).ctypes.data_as(POINTER(c_float))
#     lut_x_p = index2pos[:, 0].astype(
#         np.float32).ctypes.data_as(POINTER(c_float))
#     lut_y_p = index2pos[:, 1].astype(
#         np.float32).ctypes.data_as(POINTER(c_float))
#     lut_z_p = index2pos[:, 2].astype(
#         np.float32).ctypes.data_as(POINTER(c_float))

#     func(img_p, proj_p,
#          index1_p, index2_p, tof_val_p,
#          lut_y_p, lut_x_p, lut_z_p,
#          image.shape[0], image.shape[1], image.shape[2],
#          image.center[0], image.center[1], image.center[2],
#          tof_reso, listmode.data.size, index2pos.shape[0], image.unit_size[0], image.unit_size[2])
#     return img_out.reshape(image.shape[::-1]).transpose((1, 2, 0))
