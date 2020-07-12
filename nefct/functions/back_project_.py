from nefct import nef_class, Any
from nefct.utils import tqdm
from nefct.data.image import Image2DT, Image3DT, Image, Image3D
from nefct.data.projection import ProjectionSequence2D, ProjectionSequence3D
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
import numpy as np
import tensorflow as tf
from typing import Callable
from nefct.data.deform_para import DeformParameter
from nefct.config import TF_USER_OP_PATH

__all__ = ('BackProject3D', 'BackProject3DT', 'BackProjectFDK3DT')
# __all__ = ('BackProjectT', 'BackProject2DT', 'BackProject3DT', 'BackProject2DNT',
#            'BackProject3DNT', 'BackProjectDeform3DT')
# dist_mod_3d = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_mod.so'
# )

# dist_back_proj_3d_flat = dist_mod_3d.back_project_flat_three
# dist_back_proj_3d_cyli = dist_mod_3d.back_project_cyli_three

# dist_mod_2d = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_2d_mod.so'
# )

# dist_back_proj_2d_flat = dist_mod_2d.back_project_flat_two
# dist_back_proj_2d_cyli = dist_mod_2d.back_project_cyli_two

# pixel_mod_3d = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/ct_pixel3d_module.so'
# )

# pixel_mod_3d_flat = pixel_mod_3d.back_project_flat
# pixel_mod_3d_cyli = pixel_mod_3d.back_project_cyli

# dist_3d_deform_mod = tf.load_op_library(
#     '/home/bill52547/Github/tensorflow/bazel-bin/tensorflow/core/user_ops/dist_3d_deform_mod.so'
# )

# dist_back_proj_deform_3d_flat = dist_3d_deform_mod.back_project_flat_deform_three
# dist_back_proj_deform_3d_cyli = dist_3d_deform_mod.back_project_cyli_deform_three


@nef_class
class BackProject:

    def __call__(self, *args, **kwargs) -> Image:
        pass


# @nef_class
# class BackProject2DT(BackProjectT):
#     config: ScannerConfig2D
#     shape: list
#     unit_size: float
#     timestamps: list
#     deformer: Callable

#     def __call__(self, proj: ProjectionSequence2D) -> Image2DT:
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()

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
#         bproj_data = np.zeros(self.shape, np.float32)
#         for i in tqdm(range(len(proj.timestamps))):
#             time_ = proj.timestamps[i]
#             config['angles'] = [proj.angles[i]]
#             bproj_data_ = dist_back_proj_2d(proj.data[:, i].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data += self.deformer(bproj_data_, time_, self.timestamps)

#         return Image2DT(bproj_data * self.unit_size, [0, 0],
#                         [s * self.unit_size for s in self.shape],
#                         self.timestamps)


# @nef_class
# class BackProject2DNT(BackProjectT):
#     config: ScannerConfig2D
#     shape: list
#     unit_size: float
#     timestamps: list

#     def __call__(self, proj: ProjectionSequence2D) -> Image2DT:
#         mode = self.config.mode
#         from nefct.utils import declare_eager_execution
#         declare_eager_execution()

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
#         timestamp_size = len(self.timestamps)
#         bproj_data = np.zeros(self.shape, np.float32)
#         for i in tqdm(range(timestamp_size)):
#             time_ = self.timestamps[i]
#             inds = np.where(proj.timestamps == time_)[0]
#             config['angles'] = proj.angles[inds]
#             bproj_data_ = dist_back_proj_2d(proj.data[:, inds].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data[:, :, i] += bproj_data_

#         return Image2DT(bproj_data * self.unit_size, [0, 0],
#                         [s * self.unit_size for s in self.shape],
#                         self.timestamps)


# @nef_class
# class BackProject3DT(BackProjectT):
#     config: ScannerConfig3D
#     shape: list
#     unit_size: float
#     timestamps: list
#     deformer: Callable

#     def __call__(self, proj: ProjectionSequence3D) -> Image3DT:
#         mode = self.config.mode

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

#         if mode == 'flat':
#             dist_back_proj_3d = dist_back_proj_3d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_3d = dist_back_proj_3d_cyli

#         bproj_data = np.zeros(self.shape, np.float32)
#         for i in tqdm(range(len(proj.timestamps))):
#             time_ = proj.timestamps[i]
#             config['angles'] = [proj.angles[i]]
#             config['offsets'] = [proj.offsets[i] / self.unit_size]
#             bproj_data_ = dist_back_proj_3d(proj.data[:, :, i].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data += self.deformer(bproj_data_, time_, self.timestamps)
#         return Image3DT(bproj_data * self.unit_size, [0, 0, 0],
#                         [s * self.unit_size for s in self.shape],
#                         self.timestamps)


# @nef_class
# class BackProjectDeform3DT(BackProjectT):
#     config: ScannerConfig3D
#     shape: list
#     unit_size: float
#     timestamps: list
#     dvf: tuple

#     def __call__(self, proj: ProjectionSequence3D) -> Image3DT:
#         mode = self.config.mode

#         config = {
#             'shape': self.shape,
#             'offsets': proj.offsets / self.unit_size,
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

#         if mode == 'flat':
#             dist_back_proj_3d = dist_back_proj_deform_3d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_3d = dist_back_proj_deform_3d_cyli

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

#         bproj_data = dist_back_proj_3d(proj.data.transpose(),
#                                        ax.transpose(), ay.transpose(), az.transpose(),
#                                        bx.transpose(), by.transpose(), bz.transpose(),
#                                        cx.transpose(), cy.transpose(), cz.transpose(),
#                                        v_data, f_data,
#                                        **config).numpy().transpose()
#         return Image3D(bproj_data * self.unit_size, [0, 0, 0],
#                        [s * self.unit_size for s in self.shape],
#                        self.timestamps)


# @nef_class
# class BackProject3DNT(BackProjectT):
#     config: ScannerConfig3D
#     shape: list
#     unit_size: float
#     timestamps: list

#     def __call__(self, proj: ProjectionSequence3D) -> Image3DT:
#         mode = self.config.mode

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

#         if mode == 'flat':
#             dist_back_proj_3d = dist_back_proj_3d_flat
#             config['da'] /= self.unit_size
#             config['ai'] /= self.unit_size
#         else:
#             dist_back_proj_3d = dist_back_proj_3d_cyli
#         timestamp_size = len(self.timestamps)
#         bproj_data = np.zeros(self.shape, np.float32)
#         for i in tqdm(range(timestamp_size)):
#             time_ = self.timestamps[i]
#             inds = np.where(proj.timestamps == time_)[0]
#             config['angles'] = proj.angles[inds]
#             config['offsets'] = proj.offsets[inds] / self.unit_size
#             bproj_data_ = dist_back_proj_3d(proj.data[:, :, inds].transpose(),
#                                             **config).numpy().transpose()
#             bproj_data[:, :, :, i] += bproj_data_
#         return Image3DT(bproj_data * self.unit_size, [0, 0, 0],
#                         [s * self.unit_size for s in self.shape],
#                         self.timestamps)
new_op_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_mod2.so')

bproj_op = new_op_mod.back_project
@nef_class
class BackProject3D(BackProject):
    scanner: ScannerConfig

    def __call__(self, projection: ProjectionSequence3D, x: Image):
        angles = projection.angles
        offsets = projection.offsets
        dx = x.unit_size[0]
        nx, ny, nz = x.shape[:3]
        if projection.scanner.mode.startswith('f'):
            mode = 0
            da = projection.scanner.detector_a.unit_size / dx
            ai = projection.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = projection.scanner.detector_a.unit_size
            ai = projection.scanner.detector_a.offset
        bproj_data = bproj_op(projection=projection.data.transpose(),
                              angles=angles,
                              offsets=offsets / dx,
                              mode=mode,
                              SO=self.scanner.SAD / dx,
                              SD=self.scanner.SID / dx,
                              nx=nx,
                              ny=ny,
                              nz=nz,
                              da=da,
                              ai=ai,
                              na=self.scanner.detector_a.number,
                              db=self.scanner.detector_b.unit_size / dx,
                              bi=self.scanner.detector_b.offset / dx,
                              nb=self.scanner.detector_b.number,).numpy().transpose()
        return x.update(data=bproj_data)


mc_bproj_op = new_op_mod.mc_back_project
@nef_class
class BackProject3DT(BackProject):
    scanner: ScannerConfig
    dvf: DeformParameter

    def __call__(self, projection: ProjectionSequence3D, x: Image):
        angles = projection.angles
        offsets = projection.offsets
        timestamps = projection.timestamps
        v_data = [vf[0] for vf in timestamps]
        f_data = [vf[1] for vf in timestamps]
        dx = x.unit_size[0]
        nx, ny, nz = x.shape[:3]
        bproj_data = mc_bproj_op(projection=projection.data.transpose(),
                                 angles=angles,
                                 offsets=offsets,
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
                                 mode=0,
                                 SO=self.scanner.SAD / dx,
                                 SD=self.scanner.SID / dx,
                                 nx=nx,
                                 ny=ny,
                                 nz=nz,
                                 da=self.scanner.detector_a.unit_size / dx,
                                 ai=self.scanner.detector_a.offset / dx,
                                 na=self.scanner.detector_a.number,
                                 db=self.scanner.detector_b.unit_size / dx,
                                 bi=self.scanner.detector_b.offset / dx,
                                 nb=self.scanner.detector_b.number,).numpy().transpose()
        return x.update(data=bproj_data)


fdk_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_fdk_mod.so')

mc_bproj_fdk_op = fdk_mod.mc_back_project_fdk


@nef_class
class BackProjectFDK3DT(BackProject):
    scanner: ScannerConfig
    dvf: DeformParameter

    def __call__(self, projection: ProjectionSequence3D, x: Image):
        angles = projection.angles
        offsets = projection.offsets
        timestamps = projection.timestamps
        v_data = [vf[0] for vf in timestamps]
        f_data = [vf[1] for vf in timestamps]
        dx = x.unit_size[0]
        nx, ny, nz = x.shape[:3]
        bproj_data = mc_bproj_fdk_op(projection=projection.data.transpose(),
                                     angles=angles,
                                     offsets=offsets,
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
                                     mode=0,
                                     SO=self.scanner.SAD / dx,
                                     SD=self.scanner.SID / dx,
                                     nx=nx,
                                     ny=ny,
                                     nz=nz,
                                     da=self.scanner.detector_a.unit_size / dx,
                                     ai=self.scanner.detector_a.offset / dx,
                                     na=self.scanner.detector_a.number,
                                     db=self.scanner.detector_b.unit_size / dx,
                                     bi=self.scanner.detector_b.offset / dx,
                                     nb=self.scanner.detector_b.number,).numpy().transpose()
        return x.update(data=bproj_data)