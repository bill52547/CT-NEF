from nefct import nef_class, Any
from nefct.data.image import Image2D, Image3D, Image, Image2DT, Image3DT
from nefct.data.projection import ProjectionSequence2D, \
    ProjectionSequence3D, ProjectionSequence
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import numpy as np
import tensorflow as tf

__all__ = ('Project', 'Project2D', 'Project3D')

dist_mod_3d = tf.load_op_library(
    '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_mod.so'
)

dist_proj_3d_flat = dist_mod_3d.project_flat_three
dist_proj_3d_cyli = dist_mod_3d.project_cyli_three

dist_mod_2d = tf.load_op_library(
    '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_2d_mod.so'
)

dist_proj_2d_flat = dist_mod_2d.project_flat_two
dist_proj_2d_cyli = dist_mod_2d.project_cyli_two


@nef_class
class Project:
    def __call__(self, image: Image) -> ProjectionSequence:
        pass


@nef_class
class Project2D(Project):
    config: ScannerConfig2D
    angles: list

    def __call__(self, image: Image2D) -> ProjectionSequence2D:
        mode = self.config.mode
        from nefct.utils import declare_eager_execution
        declare_eager_execution()

        config = {
            'shape': image.shape,
            'angles': self.angles,
            'SID': self.config.SID / image.unit_size[0],
            'SAD': self.config.SAD / image.unit_size[0],
            'na': self.config.detector.number,
            'da': self.config.detector.unit_size,
            'ai': self.config.detector.offset
        }
        if mode == 'flat':
            dist_proj_2d = dist_proj_2d_flat
            config['da'] /= image.unit_size[0]
            config['ai'] /= image.unit_size[0]
        else:
            dist_proj_2d = dist_proj_2d_cyli

        proj_data = dist_proj_2d(image.data.transpose(),
                                 **config).numpy().transpose()
        return ProjectionSequence2D(proj_data * image.unit_size[0], self.config, self.angles)


@nef_class
class Project3D(Project):
    config: ScannerConfig3D
    angles: list
    offsets: list

    def __call__(self, image: Image) -> ProjectionSequence3D:
        if self.offsets is None:
            object.__setattr__(self, 'offsets', self.angles * 0)
        mode = self.config.mode

        config = {
            'shape': image.shape,
            'offsets': [off / image.unit_size[0] for off in self.offsets],
            'angles': self.angles,
            'SID': self.config.SID / image.unit_size[0],
            'SAD': self.config.SAD / image.unit_size[0],
            'na': self.config.detector_a.number,
            'da': self.config.detector_a.unit_size,
            'ai': self.config.detector_a.offset,
            'nb': self.config.detector_b.number,
            'db': self.config.detector_b.unit_size / image.unit_size[0],
            'bi': self.config.detector_b.offset / image.unit_size[0]
        }

        if mode == 'flat':
            dist_proj_3d = dist_proj_3d_flat
            config['da'] /= image.unit_size[0]
            config['ai'] /= image.unit_size[0]
        else:
            dist_proj_3d = dist_proj_3d_cyli

        proj_data = dist_proj_3d(image.data.transpose(),
                                 **config).numpy().transpose()
        return ProjectionSequence3D(proj_data * image.unit_size[0], self.config, self.angles,
                                    self.offsets)
