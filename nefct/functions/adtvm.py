import h5py
import numpy as np
import nefct as nef
from nefct.existing_geometry import *
import tensorflow as tf
import attr

tf.compat.v1.enable_eager_execution()

from tqdm import tqdm_notebook
from time import time
from nefct.functions.register import Register3D

import deepdish as dd
from nefct.functions.project import Project, Project3DT, Project3D
from nefct.functions.back_project import BackProject, BackProject3DT, BackProject3D
from nefct.functions.deform import Deform, InvertDeform
from nefct.data.projection import ProjectionSequence
from nefct.functions.sart import MCSART
from nefct.data.deform_para import DeformParameter
from nefct.data.image import Image3DT, Image3D
from nefct.base.base import nef_class
from nefct.functions.conj_grad import ConjGrad
from nefct.functions.total_variation import TotalVari, TotalVariT, InvertTotalVari, InvertTotalVariSingle3DT
from nefct.functions.soft_shrinkage import SoftShrink
from nefct.utils import tqdm
from nefct.geometry.scanner_config import ScannerConfig3D, ScannerConfig2D
from nefct.functions.binning_strategy import binning_strategy

from skimage.restoration import denoise_tv_chambolle as tv_denoise


# projector = lambda x_: Project3DT(scanner, dvf_para, angles, offsets)(x_, timestamps)
# bprojector = lambda x_: BackProject3DT(scanner, dvf_para)(x_, x)

@nef_class
class ADTVM:
    pass


@nef_class
class ADTVM_3D(ADTVM):
    scanner: ScannerConfig3D
    n_iter: int
    cg_n_iter: int
    rho: float = attr.ib(default=1)
    lamb_: float = attr.ib(default=1)

    def __call__(self, projection: ProjectionSequence, x: Image3D) -> Image3D:
        x = x + 0
        projector = Project3D(self.scanner, projection.angles, projection.offsets)
        bprojector = lambda x_: BackProject3D(self.scanner)(x_, x)

        tv3d = TotalVari()
        itv3d = InvertTotalVari()
        cg = ConjGrad(self.cg_n_iter, projector, bprojector, tv3d, itv3d, self.rho)

        if self.n_iter == 1:
            tbar = range(self.n_iter)
        else:
            tbar = tqdm(range(self.n_iter))
        u = tv3d(x)
        b = u * 0
        err_p = 10 ** 100
        for iter_ in tbar:
            d = u - b
            x = cg(projection, x, d)
            u = SoftShrink(self.lamb_ / self.rho)(tv3d(x) + b)
            b = b + tv3d(x) - u
            err = np.sum((projection.data - projector(x).data) ** 2)
            if err > err_p:
                print(iter_, err_p, err)
                break
            else:
                err_p = err
            # x_data = x.data + 0
            # x_data = tv_denoise(x_data, weight=50)
            # x = x.update(data=x_data)
        return x


# @nef_class
# class ADTVM_4D(ADTVM):
#     scanner: ScannerConfig3D
#     n_iter: int
#     cg_n_iter: int
#     lamb_: float
#     mu: float

#     def __call__(self, projection: ProjectionSequence, x: Image3DT) -> Image3DT:
#         nbin = x.shape[3]
#         v_data = np.array([time_[0] for time_ in projection.timestamps])
#         f_data = np.array([time_[1] for time_ in projection.timestamps])
#         bin_filter, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
#         total_var = TotalVari()
#         soft_sh = SoftShrink(self.lamb_)
#         d = [total_var(x[ibin]) for ibin in range(nbin)]
#         v = [total_var(x[ibin]) for ibin in range(nbin)]
#         f = [0] * nbin
#         for iter_ in tqdm(range(self.n_iter)):
#             for ibin in range(nbin):
#                 inv_total_var = InvertTotalVari()

#                 filt_ = slice(num_in_bin[ibin], num_in_bin[ibin + 1])
#                 projector = Project3D(self.scanner, 
#                                       projection.angles[bin_filter][filt_], 
#                                       projection.offsets[bin_filter][filt_])
#                 bprojector = lambda x_: BackProject3D(self.scanner)(x_, x[ibin])
#                 cg = ConjGrad(self.cg_n_iter, projector, bprojector, total_var, inv_total_var, self.mu)
#                 f[ibin] = projector(x[ibin])
#                 x.data[:,:,:,ibin] = cg(projection[bin_filter][filt_], x[ibin], d[ibin] - v[ibin]).data 
#                 f[ibin] = f[ibin] + projector(x[ibin]) - projection[bin_filter][filt_]
#             d = soft_sh(total_var(x) + v)
#             v = v + total_var(x) - d

#         return x
@nef_class
class ADTVM_4D(ADTVM):
    scanner: ScannerConfig3D
    dvf: DeformParameter
    n_iter: int
    cg_n_iter: int

    def __call__(self, projection: ProjectionSequence, x: Image3DT) -> Image3DT:
        nbin = x.shape[3]
        adtvm_3d = ADTVM_3D(self.scanner, 1, self.cg_n_iter)

        x_out = x * 1
        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_filter, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        deformer = Deform(self.dvf)
        int_deformer = InvertDeform(self.dvf)
        for iter_ in range(self.n_iter):
            for ibin in tqdm(range(nbin)):
                filt_ = slice(num_in_bin[ibin], num_in_bin[ibin + 1])
                if ibin == 0:
                    x = deformer(int_deformer(x_out[nbin - 1] * 1, v0_data[-1], f0_data[-1]), v0_data[0], f0_data[0])
                else:
                    x = deformer(int_deformer(x_out[ibin - 1] * 1, v0_data[ibin - 1], f0_data[ibin - 1]), v0_data[ibin],
                                 f0_data[ibin])
                proj_ = projection[bin_filter][filt_]
                x = adtvm_3d(proj_, x)
                x_out.data[:, :, :, ibin] = x.data
        return x_out


from nefct.functions.deform import Deform, InvertDeform


@nef_class
class ADTVM_4DDeform(ADTVM):
    scanner: ScannerConfig3D
    dvf: DeformParameter
    n_iter: int
    cg_n_iter: int
    rho: float
    lamb_: float

    def __call__(self, projection: ProjectionSequence, x: Image3D) -> Image3DT:
        nbin = x.shape[3]

        x_out = x * 1
        v_data = np.array([time_[0] for time_ in projection.timestamps])
        f_data = np.array([time_[1] for time_ in projection.timestamps])
        bin_filter, v0_data, f0_data, num_in_bin = binning_strategy(v_data, f_data, nbin)
        tv3d = TotalVari()
        itv3d = InvertTotalVari()
        err_p = 10 ** 100
        u = tv3d(x[0])
        b = u * 1
        ibin = 0
        filt_ = slice(num_in_bin[ibin], num_in_bin[ibin + 1])
        proj_ = projection[bin_filter][filt_]
        projector = Project3DT(self.scanner, self.dvf, proj_.angles, proj_.offsets, proj_.timestamps)
        bprojector = lambda x_: BackProject3DT(self.scanner, self.dvf)(x_, x)
        maxAtA = np.max(bprojector(projector(x * 0 + 1)).data)
        print('maxAtA', maxAtA)
        for iter_ in tqdm(range(self.n_iter)):
            x = x_out[ibin]
            cg = ConjGrad(self.cg_n_iter, projector, bprojector, tv3d, itv3d, self.rho)
            d = u - b
            x = cg(proj_, x, d)
            u = SoftShrink(self.lamb_ * maxAtA / self.rho)(tv3d(x) + b)
            b = b + tv3d(x) - u
            err = np.sum((projection.data[:, :, bin_filter[filt_]] - projector(x).data) ** 2)
            if err > err_p:
                print(iter_, err_p, err)
                break
            else:
                err_p = err
            x_out.data[:, :, :, ibin] = x.data
        return x_out
