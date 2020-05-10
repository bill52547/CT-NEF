from scipy.fftpack import fft, fftshift, ifft, ifftshift
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from nefct.base.base import nef_class
from nefct.data.projection import ProjectionSequence3D
from nefct.geometry import ScannerConfig3D
from nefct.data import Image3D
from nefct.utils import tqdm
from nefct.config import TF_USER_OP_PATH
fdk_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_fdk_mod.so')

bproj_op = fdk_mod.back_project_fdk

def ramp_filt(n):
    nn = np.arange(-n / 2, n / 2, 1)
    h = np.zeros(nn.size, dtype = np.float32)
    h[n // 2] = 0.25
    odd = nn % 2 == 1
    h[odd] = -1 / (np.pi * nn[odd]) ** 2
    return h

def Filter(kernel, order, d):
    f_kernel = np.abs(fft(kernel))* 2
    filt = f_kernel[:order // 2 + 1].T
    w = 2 * np.pi * np.arange(filt.size) / order
    
    filt[1:] = filt[1:] * (1 + np.cos(w[1:] / d)) / 2
    filt[w > np.pi * d] = 0
    filt = np.hstack((filt, filt[-2:0:-1]))
    return filt

def fft_(x):
    return fftshift(fft(x))

def ifft_(x):
    return np.real(ifft(ifftshift(x)))

def filtering(projection: ProjectionSequence3D, scanner: ScannerConfig3D):
    SO = scanner.SAD
    SD = scanner.SID
    da = scanner.detector_a.unit_size
    ai = scanner.detector_a.offset
    na = scanner.detector_a.number
    db = scanner.detector_b.unit_size
    bi = scanner.detector_b.offset
    nb = scanner.detector_b.number
    if not ai == 0 or not bi == 0:
        raise ValueError
    ratio = SO / SD
    nu, ui, du = na, ai * ratio, da * ratio
    nv, vi, dv = nb, bi * ratio, db * ratio
    us = (np.arange(nu) + 0.5 - nu / 2) * du + ui

    ramp_kernel = ramp_filt(nu)
    add_num = 0
    us_full = (np.arange(-add_num, nu) + 0.5 - (nu + add_num) / 2) * du + ui
    vs = (np.arange(nv) + 0.5 - nv / 2) * dv + vi
    uu, vv = np.meshgrid(us_full, vs, indexing = 'ij')
    w = SO / np.sqrt(SO ** 2 + uu ** 2 + vv ** 2)

    filt_ = np.abs(us_full)
    filt_full = np.kron(filt_, [[1]] * nb).T
    proj_data = projection.data
    proj_data2 = proj_data * 1
    fproj = np.zeros((nu, nv), dtype = np.complex128)
    for iview in tqdm(range(projection.shape[2])):
        for iv in range(nv):
            fproj[:, iv] = fft_(proj_data[:,iv,iview] * w[:, iv])
            proj_data2[:,iv,iview] = ifft_(fproj[:, iv] * filt_).astype(np.float32)

    projection_filtered = projection.update(data = proj_data2)

    return projection_filtered

@nef_class
class FDK:
    scanner: ScannerConfig3D

    def __call__(self, projection: ProjectionSequence3D, x: Image3D):
        angles = projection.angles
        offsets = projection.offsets
        timestamps = projection.timestamps
        dx = x.unit_size[0]
        nx, ny, nz = x.shape[:3]
        projection_filtered = filtering(projection, self.scanner)
        bproj_data = bproj_op(projection = projection_filtered.data.transpose(), 
                        angles = angles, 
                        offsets = offsets,
                        mode = 0,
                        SO = self.scanner.SAD / dx,
                        SD = self.scanner.SID / dx,
                        nx = nx,
                        ny = ny,
                        nz = nz,
                        da = self.scanner.detector_a.unit_size / dx,
                        ai = self.scanner.detector_a.offset / dx,
                        na = self.scanner.detector_a.number,
                        db = self.scanner.detector_b.unit_size / dx,
                        bi = self.scanner.detector_b.offset / dx,
                        nb = self.scanner.detector_b.number).numpy().transpose()
        return x.update(data = bproj_data) 

