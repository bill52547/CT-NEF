from nefct import nef_class, Any
from nefct.utils import tqdm
from nefct.data.image import Image2DT, Image3DT, Image
from nefct.data.projection import ProjectionSequence2D, ProjectionSequence3D
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import numpy as np
import tensorflow as tf
from typing import Callable

__all__ = ('BackProjectT', 'BackProject2DT', 'BackProject3DT', 'BackProject2DNT', 'BackProject3DNT')
dist_mod_3d = tf.load_op_library(
    '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_mod.so'
)

dist_back_proj_3d_flat = dist_mod_3d.back_project_flat_three
dist_back_proj_3d_cyli = dist_mod_3d.back_project_cyli_three

dist_mod_2d = tf.load_op_library(
    '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_2d_mod.so'
)

dist_back_proj_2d_flat = dist_mod_2d.back_project_flat_two
dist_back_proj_2d_cyli = dist_mod_2d.back_project_cyli_two


@nef_class
class BackProjectT:
    config: ScannerConfig

    def __call__(self, *args, **kwargs) -> Image:
        pass


@nef_class
class BackProject2DT(BackProjectT):
    config: ScannerConfig2D
    shape: list
    unit_size: float
    timestamps: list
    deformer: Callable

    def __call__(self, proj: ProjectionSequence2D) -> Image2DT:
        mode = self.config.mode
        from nefct.utils import declare_eager_execution
        declare_eager_execution()

        config = {
            'shape': self.shape,
            'angles': proj.angles,
            'SID': self.config.SID / self.unit_size,
            'SAD': self.config.SAD / self.unit_size,
            'na': self.config.detector.number,
            'da': self.config.detector.unit_size,
            'ai': self.config.detector.offset
        }
        if mode == 'flat':
            dist_back_proj_2d = dist_back_proj_2d_flat
            config['da'] /= self.unit_size
            config['ai'] /= self.unit_size
        else:
            dist_back_proj_2d = dist_back_proj_2d_cyli
        bproj_data = np.zeros(self.shape, np.float32)
        for i in tqdm(range(len(proj.timestamps))):
            time_ = proj.timestamps[i]
            config['angles'] = [proj.angles[i]]
            bproj_data_ = dist_back_proj_2d(proj.data[:, i].transpose(),
                                            **config).numpy().transpose()
            bproj_data += self.deformer(bproj_data_, time_, self.timestamps)

        return Image2DT(bproj_data * self.unit_size, [0, 0],
                        [s * self.unit_size for s in self.shape],
                        self.timestamps)


@nef_class
class BackProject2DNT(BackProjectT):
    config: ScannerConfig2D
    shape: list
    unit_size: float
    timestamps: list

    def __call__(self, proj: ProjectionSequence2D) -> Image2DT:
        mode = self.config.mode
        from nefct.utils import declare_eager_execution
        declare_eager_execution()

        config = {
            'shape': self.shape,
            'angles': proj.angles,
            'SID': self.config.SID / self.unit_size,
            'SAD': self.config.SAD / self.unit_size,
            'na': self.config.detector.number,
            'da': self.config.detector.unit_size,
            'ai': self.config.detector.offset
        }
        if mode == 'flat':
            dist_back_proj_2d = dist_back_proj_2d_flat
            config['da'] /= self.unit_size
            config['ai'] /= self.unit_size
        else:
            dist_back_proj_2d = dist_back_proj_2d_cyli
        timestamp_size = len(self.timestamps)
        bproj_data = np.zeros(self.shape, np.float32)
        for i in tqdm(range(timestamp_size)):
            time_ = self.timestamps[i]
            inds = np.where(proj.timestamps == time_)[0]
            config['angles'] = proj.angles[inds]
            bproj_data_ = dist_back_proj_2d(proj.data[:, inds].transpose(),
                                            **config).numpy().transpose()
            bproj_data[:, :, i] += bproj_data_

        return Image2DT(bproj_data * self.unit_size, [0, 0],
                        [s * self.unit_size for s in self.shape],
                        self.timestamps)


@nef_class
class BackProject3DT(BackProjectT):
    config: ScannerConfig3D
    shape: list
    unit_size: float
    timestamps: list
    deformer: Callable

    def __call__(self, proj: ProjectionSequence3D) -> Image3DT:
        mode = self.config.mode

        config = {
            'shape': self.shape,
            'offsets': proj.offsets,
            'angles': proj.angles,
            'SID': self.config.SID / self.unit_size,
            'SAD': self.config.SAD / self.unit_size,
            'na': self.config.detector_a.number,
            'da': self.config.detector_a.unit_size,
            'ai': self.config.detector_a.offset,
            'nb': self.config.detector_b.number,
            'db': self.config.detector_b.unit_size / self.unit_size,
            'bi': self.config.detector_b.offset / self.unit_size
        }

        if mode == 'flat':
            dist_back_proj_3d = dist_back_proj_3d_flat
            config['da'] /= self.unit_size
            config['ai'] /= self.unit_size
        else:
            dist_back_proj_3d = dist_back_proj_3d_cyli

        bproj_data = np.zeros(self.shape, np.float32)
        for i in tqdm(range(len(proj.timestamps))):
            time_ = proj.timestamps[i]
            config['angles'] = [proj.angles[i]]
            config['offsets'] = [proj.offsets[i] / self.unit_size]
            bproj_data_ = dist_back_proj_3d(proj.data[:, :, i].transpose(),
                                            **config).numpy().transpose()
            bproj_data += self.deformer(bproj_data_, time_, self.timestamps)
        return Image3DT(bproj_data * self.unit_size, [0, 0, 0],
                        [s * self.unit_size for s in self.shape],
                        self.timestamps)


@nef_class
class BackProject3DNT(BackProjectT):
    config: ScannerConfig3D
    shape: list
    unit_size: float
    timestamps: list

    def __call__(self, proj: ProjectionSequence3D) -> Image3DT:
        mode = self.config.mode

        config = {
            'shape': self.shape,
            'offsets': proj.offsets,
            'angles': proj.angles,
            'SID': self.config.SID / self.unit_size,
            'SAD': self.config.SAD / self.unit_size,
            'na': self.config.detector_a.number,
            'da': self.config.detector_a.unit_size,
            'ai': self.config.detector_a.offset,
            'nb': self.config.detector_b.number,
            'db': self.config.detector_b.unit_size / self.unit_size,
            'bi': self.config.detector_b.offset / self.unit_size
        }

        if mode == 'flat':
            dist_back_proj_3d = dist_back_proj_3d_flat
            config['da'] /= self.unit_size
            config['ai'] /= self.unit_size
        else:
            dist_back_proj_3d = dist_back_proj_3d_cyli
        timestamp_size = len(self.timestamps)
        bproj_data = np.zeros(self.shape, np.float32)
        for i in tqdm(range(timestamp_size)):
            time_ = self.timestamps[i]
            inds = np.where(proj.timestamps == time_)[0]
            config['angles'] = proj.angles[inds]
            config['offsets'] = proj.offsets[inds] / self.unit_size
            bproj_data_ = dist_back_proj_3d(proj.data[:, :, inds].transpose(),
                                            **config).numpy().transpose()
            bproj_data[:, :, :, i] += bproj_data_
        return Image3DT(bproj_data * self.unit_size, [0, 0, 0],
                        [s * self.unit_size for s in self.shape],
                        self.timestamps)
