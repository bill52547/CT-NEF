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
from nefct.data.image import Image3DT, Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.project import Project
from nefct.functions.back_project import BackProject
from nefct.geometry.scanner_config import ScannerConfig, ScannerConfig2D, ScannerConfig3D
from nefct.functions.project import Project3D
from nefct.functions.back_project import BackProject3D
import tensorflow as tf
from nefct.utils import tqdm
from nefct.data.deform_para import DeformParameter
from nefct.functions.binning_strategy import binning_strategy
from nefct.functions.deform_and_cal_para import deform_and_cal_para
from nefct.config import TF_USER_OP_PATH


@nef_class
class SART:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig

    def __call__(self, projection: ProjectionSequence, nbin: int, x: Image = None) -> Image:
        angles = projection.angles
        offsets = projection.offsets
        new_op_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_mod2.so')
        sart_op = new_op_mod.sart
        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_inds, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        dx = x.unit_size[0]
        if projection.scanner.mode.startswith('f'):
            mode = 0
            da = projection.scanner.detector_a.unit_size / dx
            ai = projection.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = projection.scanner.detector_a.unit_size
            ai = projection.scanner.detector_a.offset
        x_data = x.data
        for ibin in tqdm(range(nbin)):
            filt = bin_inds[num_in_bin[ibin]: num_in_bin[ibin + 1]]
            x_data[:, :, :, ibin] = sart_op(image=x_data[:, :, :, ibin].transpose(),
                                            projection=projection.data[:, :, filt].transpose(),
                                            angles=projection.angles[filt],
                                            offsets=projection.offsets[filt] / dx,
                                            mode=mode,
                                            SO=projection.scanner.SAD / dx,
                                            SD=projection.scanner.SID / dx,
                                            nx=x.shape[0],
                                            ny=x.shape[1],
                                            nz=x.shape[2],
                                            da=da,
                                            ai=ai,
                                            na=projection.scanner.detector_a.number,
                                            db=projection.scanner.detector_b.unit_size / dx,
                                            bi=projection.scanner.detector_b.offset / dx,
                                            nb=projection.scanner.detector_b.number,
                                            n_iter=self.n_iter,
                                            lamb=self.lambda_
                                            ).numpy().transpose()
        x = x.update(data=x_data)

        return x


@nef_class
class MCSART:
    out_n_iter: int
    n_iter: int
    lambda_: float
    scanner: ScannerConfig
    dvf: DeformParameter

    def __call__(self, projection: ProjectionSequence,
                 nbin: int,
                 x: Image3DT = None,
                 *,
                 is_deform=True) -> Image3DT:
        new_op_mod = tf.load_op_library(
            TF_USER_OP_PATH + '/new_op_mod.so')
        mc_sart_op = new_op_mod.mc_sart
        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_inds, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        x_data = x.data
        dx = x.unit_size[0]
        new_dvf_para = self.dvf
        if projection.scanner.mode.startswith('f'):
            mode = 0
            da = projection.scanner.detector_a.unit_size / dx
            ai = projection.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = projection.scanner.detector_a.unit_size
            ai = projection.scanner.detector_a.offset
        for i in tqdm(range(self.out_n_iter)):
            if not is_deform:
                v_data_ = v_data[bin_inds]
                f_data_ = f_data[bin_inds]
                v0_data_ = v0_data
                f0_data_ = f0_data
            else:
                v_data_ = v_data[bin_inds] - v0_data[0]
                f_data_ = f_data[bin_inds] - f0_data[0]
                v0_data_ = v0_data - v0_data[0]
                f0_data_ = f0_data - f0_data[0]
            x_data = mc_sart_op(image=x_data.transpose(),
                                projection=projection.data[:, :, bin_inds].transpose(),
                                angles=projection.angles[bin_inds],
                                offsets=projection.offsets[bin_inds] / dx,
                                ax=self.dvf.ax.transpose(),
                                ay=self.dvf.ay.transpose(),
                                az=self.dvf.az.transpose(),
                                bx=self.dvf.bx.transpose(),
                                by=self.dvf.by.transpose(),
                                bz=self.dvf.bz.transpose(),
                                cx=self.dvf.cx.transpose(),
                                cy=self.dvf.cy.transpose(),
                                cz=self.dvf.cz.transpose(),
                                v_data=v_data_,
                                f_data=f_data_,
                                v0_data=v0_data_,
                                f0_data=f0_data_,
                                num_in_bin=num_in_bin,
                                mode=mode,
                                SO=projection.scanner.SAD / dx,
                                SD=projection.scanner.SID / dx,
                                nx=x.shape[0],
                                ny=x.shape[1],
                                nz=x.shape[2],
                                da=da,
                                ai=ai,
                                na=projection.scanner.detector_a.number,
                                db=projection.scanner.detector_b.unit_size / dx,
                                bi=projection.scanner.detector_b.offset / dx,
                                nb=projection.scanner.detector_b.number,
                                n_iter=self.n_iter,
                                lamb=self.lambda_,
                                out_iter=i).numpy().transpose()
            x = x.update(data=x_data, timestamps=list(zip(v0_data, f0_data)))
            if not is_deform:
                continue
            if i < self.out_n_iter - 1 or not self.out_n_iter > 1:
                new_dvf_para = deform_and_cal_para(x)
                object.__setattr__(self, 'dvf', new_dvf_para)
        return x, new_dvf_para


@nef_class
class SMEIR:
    out_n_iter: int
    n_iter: int
    lambda_: float
    scanner: ScannerConfig
    dvf: DeformParameter

    def __call__(self, projection: ProjectionSequence,
                 nbin: int,
                 x: Image3DT = None,
                 *,
                 is_deform=True) -> Image3DT:
        new_op_mod = tf.load_op_library(
            TF_USER_OP_PATH + '/new_op_mod.so')
        mc_sart_op = new_op_mod.mc_sart
        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_inds, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        for ibin in range(nbin):
            v_data[bin_inds[num_in_bin[ibin]: num_in_bin[ibin + 1]]] = v0_data[ibin]
            f_data[bin_inds[num_in_bin[ibin]: num_in_bin[ibin + 1]]] = f0_data[ibin]
        x_data = x.data
        dx = x.unit_size[0]
        new_dvf_para = self.dvf
        if projection.scanner.mode.startswith('f'):
            mode = 0
            da = projection.scanner.detector_a.unit_size / dx
            ai = projection.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = projection.scanner.detector_a.unit_size
            ai = projection.scanner.detector_a.offset
        for i in tqdm(range(self.out_n_iter)):
            if not is_deform:
                v_data_ = v_data[bin_inds]
                f_data_ = f_data[bin_inds]
                v0_data_ = v0_data
                f0_data_ = f0_data
            else:
                v_data_ = v_data[bin_inds] - v0_data[0]
                f_data_ = f_data[bin_inds] - f0_data[0]
                v0_data_ = v0_data - v0_data[0]
                f0_data_ = f0_data - f0_data[0]
            x_data = mc_sart_op(image=x_data.transpose(),
                                projection=projection.data[:, :, bin_inds].transpose(),
                                angles=projection.angles[bin_inds],
                                offsets=projection.offsets[bin_inds] / dx,
                                ax=self.dvf.ax.transpose(),
                                ay=self.dvf.ay.transpose(),
                                az=self.dvf.az.transpose(),
                                bx=self.dvf.bx.transpose(),
                                by=self.dvf.by.transpose(),
                                bz=self.dvf.bz.transpose(),
                                cx=self.dvf.cx.transpose(),
                                cy=self.dvf.cy.transpose(),
                                cz=self.dvf.cz.transpose(),
                                v_data=v_data_,
                                f_data=f_data_,
                                v0_data=v0_data_,
                                f0_data=f0_data_,
                                num_in_bin=num_in_bin,
                                mode=mode,
                                SO=projection.scanner.SAD / dx,
                                SD=projection.scanner.SID / dx,
                                nx=x.shape[0],
                                ny=x.shape[1],
                                nz=x.shape[2],
                                da=da,
                                ai=ai,
                                na=projection.scanner.detector_a.number,
                                db=projection.scanner.detector_b.unit_size / dx,
                                bi=projection.scanner.detector_b.offset / dx,
                                nb=projection.scanner.detector_b.number,
                                n_iter=self.n_iter,
                                lamb=self.lambda_,
                                out_iter=i).numpy().transpose()
            x = x.update(data=x_data, timestamps=list(zip(v0_data, f0_data)))
            if not is_deform:
                continue
            if i < self.out_n_iter - 1 or not self.out_n_iter > 1:
                new_dvf_para = deform_and_cal_para(x)
                object.__setattr__(self, 'dvf', new_dvf_para)
        return x, new_dvf_para


def norm2(data0, data1):
    return np.sqrt(np.sum((data0 - data1) ** 2) / np.sum(data0 ** 2))


@nef_class
class SART2:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig

    def __call__(self, projection: ProjectionSequence, nbin: int, x: Image = None) -> Image:
        new_op_mod = tf.load_op_library(TF_USER_OP_PATH + '/new_op_mod.so')

        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_inds, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        dx = x.unit_size[0]
        if projection.scanner.mode.startswith('f'):
            mode = 0
            da = projection.scanner.detector_a.unit_size / dx
            ai = projection.scanner.detector_a.offset / dx
        else:
            mode = 1
            da = projection.scanner.detector_a.unit_size
            ai = projection.scanner.detector_a.offset
        x_data = x.data
        err = np.zeros((num_in_bin[-1], self.n_iter), dtype=np.float32)
        img1 = np.ones((x.shape[::-1]), dtype=np.float32)
        for ibin in tqdm(range(nbin)):
            filt = bin_inds[num_in_bin[ibin]: num_in_bin[ibin + 1]]
            for iter_ in tqdm(range(self.n_iter)):
                for iangle in tqdm(filt):
                    projector = lambda x: new_op_mod.project(image=x_data.transpose(),
                                                             angles=projection.angles[iangle],
                                                             offsets=projection.offsets[iangle] / dx,
                                                             mode=mode,
                                                             SO=projection.scanner.SAD / dx,
                                                             SD=projection.scanner.SID / dx,
                                                             nx=x.shape[0],
                                                             ny=x.shape[1],
                                                             nz=x.shape[2],
                                                             da=da,
                                                             ai=ai,
                                                             na=projection.scanner.detector_a.number,
                                                             db=projection.scanner.detector_b.unit_size / dx,
                                                             bi=projection.scanner.detector_b.offset / dx,
                                                             nb=projection.scanner.detector_b.number)

                    bprojector = lambda x: new_op_mod.back_project(projection=x.data[:, :, iangle].transpose(),
                                                                   angles=projection.angles[iangle],
                                                                   offsets=projection.offsets[iangle] / dx,
                                                                   mode=mode,
                                                                   SO=projection.scanner.SAD / dx,
                                                                   SD=projection.scanner.SID / dx,
                                                                   nx=x.shape[0],
                                                                   ny=x.shape[1],
                                                                   nz=x.shape[2],
                                                                   da=da,
                                                                   ai=ai,
                                                                   na=projection.scanner.detector_a.number,
                                                                   db=projection.scanner.detector_b.unit_size / dx,
                                                                   bi=projection.scanner.detector_b.offset / dx,
                                                                   nb=projection.scanner.detector_b.number)
                    bp = bprojector(projection.data[:, :, iangle].transpose() - projector(x_data[:, :, :, ibin]))
                    ebp = bprojector(projector(img1))
                    x_data[:, :, :, ibin] += bp / ebp
                    err[iangle, iter_] = norm2(projection.data[:, :, iangle], projector(x_data[:, :, :, ibin]))
            # x_data[:, :, :, ibin] = sart_op(image=x_data[:, :, :, ibin].transpose(),
            #                                 projection=projection.data[:, :, filt].transpose(),
            #                                 angles=projection.angles[filt],
            #                                 offsets=projection.offsets[filt] / dx,
            #                                 mode=mode,
            #                                 SO=projection.scanner.SAD / dx,
            #                                 SD=projection.scanner.SID / dx,
            #                                 nx=x.shape[0],
            #                                 ny=x.shape[1],
            #                                 nz=x.shape[2],
            #                                 da=da,
            #                                 ai=ai,
            #                                 na=projection.scanner.detector_a.number,
            #                                 db=projection.scanner.detector_b.unit_size / dx,
            #                                 bi=projection.scanner.detector_b.offset / dx,
            #                                 nb=projection.scanner.detector_b.number,
            #                                 n_iter=self.n_iter,
            #                                 lamb=self.lambda_
            #                                 ).numpy().transpose()
        x = x.update(data=x_data)

        return x, err
