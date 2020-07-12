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
class Backproject:
    scanner: ScannerConfig
    angles: np.ndarray
    offsets_b: np.ndarray
    offsets_a: np.ndarray

    def __call__(self, projection: ProjectionSequence3D, image: Image3D) -> Image3D:
        dll = ctypes.CDLL(CTYPE_SO_FILE + 'back_dd.so',
                          mode=ctypes.RTLD_GLOBAL)
        func = dll.backproject
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
        image_data = np.zeros(image.shape, dtype=np.float32).ravel()
        img_p = image_data.ravel().ctypes.data_as(POINTER(c_float))
        proj_p = projection.data.ctypes.data_as(POINTER(c_float))
        angles_p = self.angles.astype(
            np.float32).ctypes.data_as(POINTER(c_float))
        off_a_p = (self.offsets_a).astype(
            np.float32).ctypes.data_as(POINTER(c_float))
        off_b_p = (self.offsets_b).astype(
            np.float32).ctypes.data_as(POINTER(c_float))

        func(img_p, proj_p,
             angles_p, off_a_p, off_b_p,
             image.shape[0], image.shape[1], image.shape[2],
             image.unit_size[0], image.unit_size[1], image.unit_size[2],
             self.scanner.SID, self.scanner.SAD,
             self.scanner.detector_a.unit_size, na,
             self.scanner.detector_b.unit_size, nb,
             nv)

        return image.update(data=image_data)
