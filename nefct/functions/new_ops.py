import h5py
import numpy as np
import nefct as nef
from nefct.existing_geometry import *
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.enable_eager_execution(config = config)
from tqdm import tqdm_notebook, tqdm
from time import time

import deepdish as dd

from nefct.config import TF_USER_OP_PATH

new_op_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_mod2.so')
proj_op = new_op_mod.project
bproj_op = new_op_mod.back_project
sart_op = new_op_mod.sart

mc_proj_op = new_op_mod.mc_project
mc_bproj_op = new_op_mod.mc_back_project
mc_sart_op = new_op_mod.mc_sart

deform_mod = tf.load_op_library(TF_USER_OP_PATH + '/tf_deform_tex_module.so'
)
deform_op = deform_mod.deform
# deform_op = deform_mod.deform
deform_invert_op = deform_mod.deform_invert


@nef.nef_class
class NewDeformer:
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray

    def __call__(self, img, v, f, p):
        mx = self.ax * v + self.bx * f + self.cx * p
        my = self.ay * v + self.by * f + self.cy * p
        mz = self.az * v + self.bz * f + self.cz * p
        nx, ny, nz = img.shape
        return deform_op(img.transpose(),
                         mx.transpose(),
                         my.transpose(),
                         mz.transpose(),
                         nx = nx,
                         ny = ny,
                         nz = nz).numpy().transpose()


@nef.nef_class
class NewInvertDeformer:
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray

    def __call__(self, img, v, f, p):
        mx = self.ax * v + self.bx * f + self.cx * p
        my = self.ay * v + self.by * f + self.cy * p
        mz = self.az * v + self.bz * f + self.cz * p
        nx, ny, nz = img.shape
        return deform_invert_op(img.transpose(),
                                mx.transpose(),
                                my.transpose(),
                                mz.transpose(),
                                nx = nx,
                                ny = ny,
                                nz = nz).numpy().transpose()


@nef.nef_class
class NewProjector:
    angles: np.ndarray
    offsets: np.ndarray
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int

    def __call__(self, img):
        proj = proj_op(image = img.transpose(),
                       angles = self.angles,
                       offsets = self.offsets,
                       mode = self.mode,
                       SO = self.SO,
                       SD = self.SD,
                       nx = self.nx,
                       ny = self.ny,
                       nz = self.nz,
                       da = self.da,
                       ai = self.ai,
                       na = self.na,
                       db = self.db,
                       bi = self.bi,
                       nb = self.nb).numpy().transpose()
        return proj


@nef.nef_class
class NewBackProjector:
    angles: np.ndarray
    offsets: np.ndarray
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int

    def __call__(self, proj: np.ndarray):
        bproj = bproj_op(projection = proj.transpose(),
                         angles = self.angles,
                         offsets = self.offsets,
                         mode = self.mode,
                         SO = self.SO,
                         SD = self.SD,
                         nx = self.nx,
                         ny = self.ny,
                         nz = self.nz,
                         da = self.da,
                         ai = self.ai,
                         na = self.na,
                         db = self.db,
                         bi = self.bi,
                         nb = self.nb).numpy().transpose()
        return bproj


@nef.nef_class
class NewSart:
    emap: np.ndarray
    angles: np.ndarray
    offsets: np.ndarray
    lamb: float
    n_iter: int
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int

    def __call__(self, proj: np.ndarray, image: np.ndarray):
        recon_img = sart_op(image = image.transpose(),
                            projection = proj.transpose(),
                            emap = self.emap.transpose(),
                            angles = self.angles,
                            offsets = self.offsets,
                            n_iter = self.n_iter,
                            lamb = self.lamb,
                            mode = self.mode,
                            SO = self.SO,
                            SD = self.SD,
                            nx = self.nx,
                            ny = self.ny,
                            nz = self.nz,
                            da = self.da,
                            ai = self.ai,
                            na = self.na,
                            db = self.db,
                            bi = self.bi,
                            nb = self.nb).numpy().transpose()
        return recon_img


@nef.nef_class
class McProjector:
    angles: np.ndarray
    offsets: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    v_data: np.ndarray
    f_data: np.ndarray
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int
    v0: float
    f0: float

    def __call__(self, img):
        proj = mc_proj_op(image = img.transpose(),
                          angles = self.angles,
                          offsets = self.offsets,
                          ax = self.ax.transpose(),
                          ay = self.ay.transpose(),
                          az = self.az.transpose(),
                          bx = self.bx.transpose(),
                          by = self.by.transpose(),
                          bz = self.bz.transpose(),
                          cx = self.cx.transpose(),
                          cy = self.cy.transpose(),
                          cz = self.cz.transpose(),
                          v_data = self.v_data,
                          f_data = self.f_data,
                          mode = self.mode,
                          SO = self.SO,
                          SD = self.SD,
                          nx = self.nx,
                          ny = self.ny,
                          nz = self.nz,
                          da = self.da,
                          ai = self.ai,
                          na = self.na,
                          db = self.db,
                          bi = self.bi,
                          nb = self.nb,
                          v0 = self.v0,
                          f0 = self.f0).numpy().transpose()
        return proj


@nef.nef_class
class McBackProjector:
    angles: np.ndarray
    offsets: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    v_data: np.ndarray
    f_data: np.ndarray
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int
    v0: float
    f0: float

    def __call__(self, proj: np.ndarray):
        bproj = mc_bproj_op(projection = proj.transpose(),
                            angles = self.angles,
                            offsets = self.offsets,
                            ax = self.ax.transpose(),
                            ay = self.ay.transpose(),
                            az = self.az.transpose(),
                            bx = self.bx.transpose(),
                            by = self.by.transpose(),
                            bz = self.bz.transpose(),
                            cx = self.cx.transpose(),
                            cy = self.cy.transpose(),
                            cz = self.cz.transpose(),
                            v_data = self.v_data,
                            f_data = self.f_data,
                            mode = self.mode,
                            SO = self.SO,
                            SD = self.SD,
                            nx = self.nx,
                            ny = self.ny,
                            nz = self.nz,
                            da = self.da,
                            ai = self.ai,
                            na = self.na,
                            db = self.db,
                            bi = self.bi,
                            nb = self.nb,
                            v0 = self.v0,
                            f0 = self.f0).numpy().transpose()
        return bproj


@nef.nef_class
class McSART:
    emap: np.ndarray
    angles: np.ndarray
    offsets: np.ndarray
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    v_data: np.ndarray
    f_data: np.ndarray
    lamb: float
    n_iter: int
    mode: int
    SO: float
    SD: float
    nx: int
    ny: int
    nz: int
    da: float
    ai: float
    na: int
    db: float
    bi: float
    nb: int
    v0: float
    f0: float

    def __call__(self, proj: np.ndarray, img: np.ndarray):
        recon_img = mc_sart_op(image = img.transpose(),
                               projection = proj.transpose(),
                               emap = self.emap.transpose(),
                               angles = self.angles,
                               offsets = self.offsets,
                               ax = self.ax.transpose(),
                               ay = self.ay.transpose(),
                               az = self.az.transpose(),
                               bx = self.bx.transpose(),
                               by = self.by.transpose(),
                               bz = self.bz.transpose(),
                               cx = self.cx.transpose(),
                               cy = self.cy.transpose(),
                               cz = self.cz.transpose(),
                               v_data = self.v_data,
                               f_data = self.f_data,
                               n_iter = self.n_iter,
                               lamb = self.lamb,
                               mode = self.mode,
                               SO = self.SO,
                               SD = self.SD,
                               nx = self.nx,
                               ny = self.ny,
                               nz = self.nz,
                               da = self.da,
                               ai = self.ai,
                               na = self.na,
                               db = self.db,
                               bi = self.bi,
                               nb = self.nb,
                               v0 = self.v0,
                               f0 = self.f0).numpy().transpose()
        return recon_img
