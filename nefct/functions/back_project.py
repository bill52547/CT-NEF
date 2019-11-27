from nefct import nef_class, Any
from nefct.utils import tqdm
from nefct.data.image import Image2D, Image3D, Image
from nefct.data.projection import ProjectionSequence2D, ProjectionSequence3D
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import numpy as np
import tensorflow as tf


__all__ = ('BackProject', 'BackProject2D', 'BackProject3D')
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

pixel_mod_3d = tf.load_op_library(
    '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/ct_pixel3d_module.so'
)

pixel_mod_3d_flat = pixel_mod_3d.back_project_flat
pixel_mod_3d_cyli = pixel_mod_3d.back_project_cyli

@nef_class
class BackProject:
    config: ScannerConfig

    def __call__(self, *args, **kwargs) -> Image:
        pass


@nef_class
class BackProject2D(BackProject):
    config: ScannerConfig2D
    shape: list
    unit_size: float

    def __call__(self, proj: ProjectionSequence2D) -> Image2D:
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

        bproj_data = dist_back_proj_2d(proj.data.transpose(),
                                       **config).numpy().transpose()
        return Image2D(bproj_data * self.unit_size, [0, 0],
                       [s * self.unit_size for s in self.shape])
#
#
# @nef_class
# class BackProject2DT(BackProject):
#     config: ScannerConfig2D
#     shape: list
#     unit_size: float
#     timestamps: list
#     deformer: Any
#
#     def __call__(self, proj: ProjectionSequence2D) -> Image2D:
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()
#
#         config = {
#             'shape': self.shape,
#             'angles': proj.angles,
#             'SID': self.config.SID,
#             'SAD': self.config.SAD,
#             'na': self.config.detector.number,
#             'da': self.config.detector.unit_size,
#             'ai': self.config.detector.offset
#         }
#         if mode == 'flat':
#             dist_back_proj_2d = dist_back_proj_2d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_2d = dist_back_proj_2d_cyli
#         bproj_data = np.zeros(self.shape, np.float32)
#         for i, time_ in enumerate(proj.timestamps):
#             config['angles'] = [proj.angles[i]]
#             bproj_data_ = dist_back_proj_2d(proj.data[:, i].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data += self.deformer(bproj_data_, time_, self.timestamps)
#
#         return Image2D(bproj_data * self.unit_size, [0, 0],
#                        [s * self.unit_size for s in self.shape])
#
#
# @nef_class
# class BackProject2DNT(BackProject):
#     config: ScannerConfig2D
#     shape: list
#     unit_size: float
#     timestamps: list
#
#     def __call__(self, proj: ProjectionSequence2D) -> list:
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()
#
#         config = {
#             'shape': self.shape,
#             'angles': proj.angles,
#             'SID': self.config.SID / self.unit_size,
#             'SAD': self.config.SAD / self.unit_size,
#             'na': self.config.detector.number,
#             'da': self.config.detector.unit_size,
#             'ai': self.config.detector.offset
#         }
#         if mode == 'flat':
#             dist_back_proj_2d = dist_back_proj_2d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_2d = dist_back_proj_2d_cyli
#         tmp_size = self.timestamps.size
#         bproj_data = [np.zeros(self.shape, np.float32)] * tmp_size
#         for i, time_ in enumerate(self.timestamps):
#             inds = np.where(proj.timestamps == time_)[0]
#             config['angles'] = proj.angles[inds]
#             bproj_data_ = dist_back_proj_2d(proj.data[:, inds].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data[i] += bproj_data_
#
#         return [Image2D(bproj_data[i] * self.unit_size, [0, 0],
#                         [s * self.unit_size for s in self.shape]) for i in range(tmp_size)]


@nef_class
class BackProject3D(BackProject):
    config: ScannerConfig3D
    shape: list
    unit_size: float

    def __call__(self, proj: ProjectionSequence3D) -> Image3D:
        mode = self.config.mode

        config = {
            'shape': self.shape,
            'offsets': [off / self.unit_size for off in proj.offsets],
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

        bproj_data = dist_back_proj_3d(proj.data.transpose(),
                                       **config).numpy().transpose()
        return Image3D(bproj_data * self.unit_size, [0, 0, 0],
                       [s * self.unit_size for s in self.shape])

#
# @nef_class
# class BackProject3DT(BackProject):
#     config: ScannerConfig3D
#     shape: list
#     unit_size: float
#     timestamps: list
#     deformer: Any
#
#     def __call__(self, proj: ProjectionSequence3D) -> Image3D:
#         mode = self.config.mode
#
#         config = {
#             'shape': self.shape,
#             'offsets': proj.offsets,
#             'angles': proj.angles,
#             'SID': self.config.SID / self.unit_size,
#             'SAD': self.config.SAD / self.unit_size,
#             'na': self.config.detector_a.number,
#             'da': self.config.detector_a.unit_size,
#             'ai': self.config.detector_a.offset,
#             'nb': self.config.detector_b.number,
#             'db': self.config.detector_b.unit_size / self.unit_size,
#             'bi': self.config.detector_b.offset / self.unit_size
#         }
#
#         if mode == 'flat':
#             dist_back_proj_3d = dist_back_proj_3d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_3d = dist_back_proj_3d_cyli
#
#         bproj_data = np.zeros(self.shape, np.float32)
#         for i in tqdm(range(len(proj.timestamps))):
#             time_ = proj.timestamps[i]
#             config['angles'] = [proj.angles[i]]
#             config['offsets'] = [proj.offsets[i] / self.unit_size]
#             bproj_data_ = dist_back_proj_3d(proj.data[:, :, i].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data += self.deformer(bproj_data_, time_, self.timestamps)
#         return Image3D(bproj_data * self.unit_size, [0, 0, 0],
#                        [s * self.unit_size for s in self.shape])
