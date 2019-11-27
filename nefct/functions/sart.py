# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: sart.py
@date: 8/28/2019
@desc:
'''

from nefct import nef_class
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.project import Project
from nefct.functions.back_project import BackProject
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
from nefct.functions import Project3D, Project3DT, BackProject3D, ProjectDeform3DT, \
    BackProjectDeform3DT
import tensorflow as tf
from nefct.utils import tqdm


@nef_class
class SART:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig
    emap: Image

    def __call__(self, projection: ProjectionSequence, x: Image = None) -> Image:
        angles = projection.angles
        offsets = projection.offsets
        projector = Project3D(self.scanner, angles, offsets)
        bprojector = BackProject3D(self.scanner, self.emap.shape, self.emap.unit_size[0])
        if x is None:
            x = self.emap * 0
        for _ in tqdm(range(self.n_iter)):
            _projection_tf = projector(x)
            _bproj_tf = bprojector(projection - _projection_tf)
            _bproj_tf2 = _bproj_tf.update(data = tf.div_no_nan(_bproj_tf.data,
                                                               self.emap.data))
            x = x.update(data = (x + _bproj_tf2 * self.lambda_).data.numpy())

        return x


@nef_class
class MCSART:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig
    emap: Image
    dvf: tuple

    def __call__(self, projection: ProjectionSequence, x: Image = None) -> Image:
        angles = projection.angles
        offsets = projection.offsets
        timestamps = projection.timestamps
        projector = ProjectDeform3DT(self.scanner, angles, offsets, timestamps, self.dvf)
        bprojector = BackProjectDeform3DT(self.scanner, self.emap.shape, self.emap.unit_size[0],
                                          timestamps, self.dvf)
        if x is None:
            x = self.emap * 0
        for _ in tqdm(range(self.n_iter)):
            _projection_tf = projector(x)
            _bproj_tf = bprojector(projection - _projection_tf)
            _bproj_tf2 = _bproj_tf.update(data = tf.div_no_nan(_bproj_tf.data,
                                                               self.emap.data))
            x = x.update(data = (x + _bproj_tf2 * self.lambda_).data.numpy())

        return x


@nef_class
class MCSART2:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig
    emap: Image
    dvf: list

    def __call__(self, projection: ProjectionSequence, ind_bin: np.ndarray,
                 x: Image = None) -> Image:
        angles = projection.angles
        offsets = projection.offsets
        timestamps = projection.timestamps
        n_bin = np.unique(ind_bin).size
        v = np.array([ts[0] for ts in timestamps])
        f = np.array([ts[1] for ts in timestamps])
        m_v = np.zeros(n_bin, dtype = np.float32)
        m_f = np.zeros(n_bin, dtype = np.float32)
        for i in range(n_bin):
            m_v[i] = np.mean(v[ind_bin == i])
            m_f[i] = np.mean(f[ind_bin == i])

        if x is None:
            x = [self.emap * 0] * n_bin
        for _ in tqdm(range(self.n_iter)):
            for ibin in range(n_bin):
                ax, ay, az, bx, by, bz = self.dvf[ibin][:6]
                if ibin > 0:
                    v_d = m_v[ibin] - m_v[ibin - 1]
                    f_d = m_f[ibin] - m_f[ibin - 1]
                    x_ = x[ibin - 1].deform(v_d * ax + f_d * bx,
                                            v_d * ay + f_d * by,
                                            v_d * az + f_d * bz)
                else:
                    v_d = m_v[ibin] - m_v[n_bin - 1]
                    f_d = m_f[ibin] - m_f[n_bin - 1]
                    x_ = x[n_bin - 1].deform(v_d * ax + f_d * bx,
                                             v_d * ay + f_d * by,
                                             v_d * az + f_d * bz)
                inds = np.where(ind_bin == ibin)[0]

                new_timestamps = [(v_ - m_v[ibin], f_ - m_f[ibin]) for (v_, f_) in timestamps[inds]]

                projector = ProjectDeform3DT(self.scanner, angles[inds], offsets[inds],
                                             new_timestamps, self.dvf[ibin])
                bprojector = BackProjectDeform3DT(self.scanner, self.emap.shape,
                                                  self.emap.unit_size[0],
                                                  new_timestamps, self.dvf[ibin])
                _projection_tf = projector(x_)
                _bproj_tf = bprojector(projection - _projection_tf)
                _bproj_tf2 = _bproj_tf.update(data = tf.div_no_nan(_bproj_tf.data,
                                                                   self.emap.data))
                x[ibin] = x[ibin].update(data = (x + _bproj_tf2 * self.lambda_).data.numpy())

        return x
