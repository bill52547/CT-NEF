from ctypes import *
import ctypes
import numpy as np
from nefct import nef_class, Any
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.geometry.scanner_config import ScannerConfig
import numpy as np
from nefct.utils import tqdm
from typing import Callable
from nefct.data.deform_para import DeformParameter
import attr
CTYPE_SO_FILE = '/home/bill52547/Workspace/NefCT/nefct/ctypes/'


@nef_class
class Project:
    scanner: ScannerConfig
    angles: np.ndarray
    offsets_a: np.ndarray
    offsets_b: np.ndarray

    def __attrs_post_init__(self):
        if self.offsets_a is None:
            object.__setattr__(self, 'offsets_a', np.zeros(
                self.angles.size, dtype=np.float32))
        if self.offsets_b is None:
            object.__setattr__(self, 'offsets_b', np.zeros(
                self.angles.size, dtype=np.float32))

    def __call__(self, image: Image) -> ProjectionSequence:
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
        off_a_p = (self.offsets_a + self.scanner.detector_a.offset).astype(
            np.float32).ctypes.data_as(POINTER(c_float))
        off_b_p = (self.offsets_b + self.scanner.detector_b.offset).astype(
            np.float32).ctypes.data_as(POINTER(c_float))

        func(proj_p, img_p,
             angles_p, off_a_p, off_b_p,
             image.shape[0], image.shape[1], image.shape[2],
             image.unit_size[0], image.unit_size[1], image.unit_size[2],
             self.scanner.SID, self.scanner.SAD,
             self.scanner.detector_a.unit_size, na,
             self.scanner.detector_b.unit_size, nb,
             nv)

        return ProjectionSequence(proj_data, self.scanner, self.angles, self.offsets_b)
