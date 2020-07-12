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
from nefct.config import TF_USER_OP_PATH

__all__ = ('Project3D', 'Project3DT')

# dist_mod_3d = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_mod.so'
# )

# dist_proj_3d_flat = dist_mod_3d.project_flat_three
# dist_proj_3d_cyli = dist_mod_3d.project_cyli_three

# dist_mod_2d = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_2d_mod.so'
# )

# dist_proj_2d_flat = dist_mod_2d.project_flat_two
# dist_proj_2d_cyli = dist_mod_2d.project_cyli_two

# dist_3d_deform_mod = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_deform_mod.so'
# )

# dist_proj_deform_3d_flat = dist_3d_deform_mod.project_flat_deform_three
# dist_proj_deform_3d_cyli = dist_3d_deform_mod.project_cyli_deform_three

new_op_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_mod2.so')

mc_proj_op = new_op_mod.mc_project
proj_op = new_op_mod.project


@nef_class
class Project:
    def __call__(self, *args, **kwargs):
        pass


# @nef_class
# class Project2DT(ProjectT):
#     config: ScannerConfig2D
#     angles: list
#     timestamps: list
#     deformer: Callable

#     def __call__(self, image: Image2D) -> ProjectionSequence2D:
#         if self.timestamps is None:
#             object.__setattr__(self, 'timestamps', self.angles * 0)
#         mode = self.config.mode

#         config = {
#             'shape': image.shape,
#             'angles': self.angles,
#             'SID': self.config.SID / image.unit_size[0],
#             'SAD': self.config.SAD / image.unit_size[0],
#             'na': self.config.detector.number,
#             'da': self.config.detector.unit_size,
#             'ai': self.config.detector.offset
#         }
#         if mode == 'flat':
#             dist_proj_2d = dist_proj_2d_flat
#             config['da'] /= image.unit_size[0]
#             config['ai'] /= image.unit_size[0]
#         else:
#             dist_proj_2d = dist_proj_2d_cyli
#         proj_data = np.zeros((self.config.detector.number, len(self.angles)), np.float32)
#         for i in tqdm(range(len(self.angles))):
#             time_ = self.timestamps[i]
#             image_ = self.deformer(image, time_)
#             config['angles'] = [self.angles[i]]
#             proj_data_ = dist_proj_2d(image_.data.transpose(),
#                                       **config).numpy().transpose()
#             proj_data[:, i] = proj_data_[:, 0]

#         return ProjectionSequence2D(proj_data * image.unit_size[0], self.config, self.angles,
#                                     self.timestamps)


# @nef_class
# class Project2DNT(ProjectT):
#     config: ScannerConfig2D
#     angles: list
#     timestamps: list

#     def __call__(self, image: Image2DT) -> ProjectionSequence2D:
#         if self.timestamps is None:
#             object.__setattr__(self, 'timestamps', self.angles * 0)
#         mode = self.config.mode

#         config = {
#             'shape': image.shape,
#             'angles': self.angles,
#             'SID': self.config.SID / image.unit_size[0],
#             'SAD': self.config.SAD / image.unit_size[0],
#             'na': self.config.detector.number,
#             'da': self.config.detector.unit_size,
#             'ai': self.config.detector.offset
#         }
#         if mode == 'flat':
#             dist_proj_2d = dist_proj_2d_flat
#             config['da'] /= image.unit_size[0]
#             config['ai'] /= image.unit_size[0]
#         else:
#             dist_proj_2d = dist_proj_2d_cyli
#         proj_data = np.zeros((self.config.detector.number, len(self.angles)), np.float32)
#         for i in tqdm(range(len(image.timestamps))):
#             time_ = image.timestamps[i]
#             image_ = image[i]
#             config['angles'] = [self.angles[self.timestamps == time_]]
#             proj_data_ = dist_proj_2d(image_.data.transpose(),
#                                       **config).numpy().transpose()
#             proj_data[:, self.timestamps == time_] = proj_data_

#         return ProjectionSequence2D(proj_data * image.unit_size[0], self.config, self.angles,
#                                     self.timestamps)
@nef_class
class Project3D(Project):
    scanner: ScannerConfig
    angles: np.ndarray
    offsets: np.ndarray

    def __call__(self, image: Image3D) -> ProjectionSequence3D:
        img_data = image.data
        dx = image.unit_size[0]
        nx, ny, nz = image.shape[:3]
        proj_data = proj_op(image=img_data.transpose(),
                            angles=self.angles,
                            offsets=self.offsets / dx,
                            ai=self.scanner.detector_a.offset / dx,
                            mode=0,
                            SO=self.scanner.SAD / dx,
                            SD=self.scanner.SID / dx,
                            nx=nx,
                            ny=ny,
                            nz=nz,
                            da=self.scanner.detector_a.unit_size / dx,
                            na=self.scanner.detector_a.number,
                            db=self.scanner.detector_b.unit_size / dx,
                            bi=self.scanner.detector_b.offset / dx,
                            nb=self.scanner.detector_b.number,).numpy().transpose()
        return ProjectionSequence3D(proj_data,
                                    self.scanner,
                                    self.angles,
                                    self.offsets
                                    )


@nef_class
class Project3DT(Project):
    scanner: ScannerConfig
    dvf: DeformParameter
    angles: np.ndarray
    offsets: np.ndarray
    timestamps: list

    def __call__(self, image: Image) -> ProjectionSequence3D:
        v_data = [vf[0] for vf in self.timestamps]
        f_data = [vf[1] for vf in self.timestamps]
        img_data = image.data
        dx = image.unit_size[0]
        nx, ny, nz = image.shape[:3]
        if self.scanner.mode.startswith('f'):
            mode = 0
            da = self.scanner.detector_a.unit_size / dx
            ai = self.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = self.scanner.detector_a.unit_size
            ai = self.scanner.detector_a.offset
        proj_data = mc_proj_op(image=img_data.transpose(),
                               angles=self.angles,
                               offsets=self.offsets,
                               ai=self.scanner.detector_a.offset / dx,
                               ax=self.dvf.ax.transpose(),
                               ay=self.dvf.ay.transpose(),
                               az=self.dvf.az.transpose(),
                               bx=self.dvf.bx.transpose(),
                               by=self.dvf.by.transpose(),
                               bz=self.dvf.bz.transpose(),
                               cx=self.dvf.cx.transpose(),
                               cy=self.dvf.cy.transpose(),
                               cz=self.dvf.cz.transpose(),
                               v_data=v_data,
                               f_data=f_data,
                               mode=mode,
                               SO=self.scanner.SAD / dx,
                               SD=self.scanner.SID / dx,
                               nx=nx,
                               ny=ny,
                               nz=nz,
                               da=da,
                               na=self.scanner.detector_a.number,
                               db=self.scanner.detector_b.unit_size / dx,
                               bi=self.scanner.detector_b.offset / dx,
                               nb=self.scanner.detector_b.number,).numpy().transpose()
        return ProjectionSequence3D(proj_data,
                                    self.scanner,
                                    self.angles,
                                    self.offsets,
                                    self.timestamps)

# @nef_class
# class Project3DT(ProjectT):
#     config: ScannerConfig3D
#     angles: list
#     offsets: list
#     timestamps: list
#     deformer: Callable

#     def __call__(self, image: Image3D) -> ProjectionSequence3D:
#         if self.timestamps is None:
#             object.__setattr__(self, 'timestamps', self.angles * 0)
#         if self.offsets is None:
#             object.__setattr__(self, 'offsets', self.angles * 0)
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()

#         config = {
#             'shape': image.shape,
#             'offsets': [off / image.unit_size[0] for off in self.offsets],
#             'angles': self.angles,
#             'SID': self.config.SID / image.unit_size[0],
#             'SAD': self.config.SAD / image.unit_size[0],
#             'na': self.config.detector_a.number,
#             'da': self.config.detector_a.unit_size,
#             'ai': self.config.detector_a.offset,
#             'nb': self.config.detector_b.number,
#             'db': self.config.detector_b.unit_size / image.unit_size[0],
#             'bi': self.config.detector_b.offset / image.unit_size[0]
#         }
#         if mode == 'flat':
#             dist_proj_3d = dist_proj_3d_flat
#             config['da'] /= image.unit_size[0]
#             config['ai'] /= image.unit_size[0]
#         else:
#             dist_proj_3d = dist_proj_3d_cyli
#         proj_data = np.zeros((self.config.detector_a.number,
#                               self.config.detector_b.number,
#                               len(self.angles)), np.float32)
#         from nefct.utils import tqdm
#         import tensorflow as tf
#         image_data_tf = image.update(data = tf.constant(image.data))
#         for i in tqdm(range(len(self.angles))):
#             time_ = self.timestamps[i]
#             image_ = self.deformer(image_data_tf, time_)
#             config['angles'] = [self.angles[i]]
#             config['offsets'] = [self.offsets[i] / image.unit_size[0]]
#             proj_data[:, :, i] = dist_proj_3d(tf.transpose(image_.data),
#                                               **config).numpy().transpose()[:, :, 0]

#         return ProjectionSequence3D(proj_data * image.unit_size[0], self.config, self.angles,
#                                     self.offsets,
#                                     self.timestamps)


# @nef_class
# class ProjectDeform3DT(ProjectT):
#     config: ScannerConfig3D
#     angles: list
#     offsets: list
#     timestamps: list
#     dvf: tuple
#     deformer: Callable

#     def __call__(self, image: Image3D) -> ProjectionSequence3D:
#         if self.timestamps is None:
#             object.__setattr__(self, 'timestamps', self.angles * 0)
#         if self.offsets is None:
#             object.__setattr__(self, 'offsets', self.angles * 0)
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()

#         config = {
#             'shape': image.shape,
#             'offsets': [off / image.unit_size[0] for off in self.offsets],
#             'angles': self.angles,
#             'SID': self.config.SID / image.unit_size[0],
#             'SAD': self.config.SAD / image.unit_size[0],
#             'na': self.config.detector_a.number,
#             'da': self.config.detector_a.unit_size,
#             'ai': self.config.detector_a.offset,
#             'nb': self.config.detector_b.number,
#             'db': self.config.detector_b.unit_size / image.unit_size[0],
#             'bi': self.config.detector_b.offset / image.unit_size[0]
#         }
#         if mode == 'flat':
#             dist_proj_3d = dist_proj_deform_3d_flat
#             config['da'] /= image.unit_size[0]
#             config['ai'] /= image.unit_size[0]
#         else:
#             dist_proj_3d = dist_proj_deform_3d_cyli
#         ax, ay, az, bx, by, bz = self.dvf[:6]
#         if isinstance(ax, Image):
#             ax = ax.data
#             ay = ay.data
#             az = az.data
#             bx = bx.data
#             by = by.data
#             bz = bz.data
#         if len(self.dvf) == 6:
#             cx = cy = cz = np.zeros(self.shape, dtype = np.float32)
#         else:
#             cx, cy, cz = self.dvf[6:]
#             if isinstance(cx, Image):
#                 cx = cx.data
#                 cy = cy.data
#                 cz = cz.data
#         v_data = [t[0] for t in self.timestamps]
#         f_data = [t[1] for t in self.timestamps]

#         proj_data = dist_proj_3d(image.data.transpose(),
#                                  ax.transpose(), ay.transpose(), az.transpose(),
#                                  bx.transpose(), by.transpose(), bz.transpose(),
#                                  cx.transpose(), cy.transpose(), cz.transpose(),
#                                  v_data, f_data,
#                                  **config).numpy().transpose()
#         return ProjectionSequence3D(proj_data * image.unit_size[0],
#                                     self.config, self.angles,
#                                     self.offsets,
#                                     self.timestamps)


# @nef_class
# class Project3DNT(ProjectT):
#     config: ScannerConfig3D
#     angles: list
#     offsets: list
#     timestamps: list

#     def __call__(self, image: Image3DT) -> ProjectionSequence3D:
#         if self.timestamps is None:
#             object.__setattr__(self, 'timestamps', self.angles * 0)
#         if self.offsets is None:
#             object.__setattr__(self, 'offsets', self.angles * 0)
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()

#         config = {
#             'shape': image.shape,
#             'offsets': [off / image.unit_size[0] for off in self.offsets],
#             'angles': self.angles,
#             'SID': self.config.SID / image.unit_size[0],
#             'SAD': self.config.SAD / image.unit_size[0],
#             'na': self.config.detector_a.number,
#             'da': self.config.detector_a.unit_size,
#             'ai': self.config.detector_a.offset,
#             'nb': self.config.detector_b.number,
#             'db': self.config.detector_b.unit_size / image.unit_size[0],
#             'bi': self.config.detector_b.offset / image.unit_size[0]
#         }
#         if mode == 'flat':
#             dist_proj_3d = dist_proj_3d_flat
#             config['da'] /= image.unit_size[0]
#             config['ai'] /= image.unit_size[0]
#         else:
#             dist_proj_3d = dist_proj_3d_cyli
#         proj_data = np.zeros((self.config.detector_a.number,
#                               self.config.detector_b.number,
#                               len(self.angles)), np.float32)
#         for i in tqdm(range(len(image.timestamps))):
#             time_ = image.timestamps[i]
#             image_ = image[i]
#             config['angles'] = self.angles[self.timestamps == time_]
#             config['offsets'] = self.offsets[self.timestamps == time_] / image.unit_size[0]

#             proj_data[:, :, self.timestamps == time_] = dist_proj_3d(tf.transpose(image_.data),
#                                                                      **config).numpy().transpose()

#         return ProjectionSequence3D(proj_data * image.unit_size[0], self.config, self.angles,
#                                     self.offsets,
#                                     self.timestamps)