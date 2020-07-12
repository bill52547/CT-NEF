from scipy.sparse.linalg import lsqr
import numpy as np
from nefct.data.deform_para import DeformParameter
from nefct.data.image import Image
from nefct.functions.register import Register3D
from nefct.utils import tqdm
from multiprocessing import Pool
import os
import nefct as nef


def register_multi(x0, x, i):
    print(os.getpid())
    dvf = Register3D()(x0, x)
    return dvf


def deform_and_cal_para(images: Image3DT):

    nbin = len(images)
    img0 = images[0]
    nx, ny, nz = img0.shape
    dvf_dct = {}
    # t_dct = {}
    # p = Pool(nbin - 1)
    # for i in range(1, nbin):
    #     p.apply_async(register_multi, args=(img0, images[i], i))
    # p.close()
    # p.join()
    # for i in tqdm(range(1, nbin)):
    #     dvf_x = nef.load(nef.Image3D, f'dvf_x_{i}.hdf5')
    #     dvf_y = nef.load(nef.Image3D, f'dvf_y_{i}.hdf5')
    #     dvf_z = nef.load(nef.Image3D, f'dvf_z_{i}.hdf5')

    #     dvf_dct[i] = [dvf_x, dvf_y, dvf_z]
    for i in tqdm(range(1, nbin)):
        dvf_dct[i] = register_multi(img0, images[i], i)
    ax_new = np.zeros((nx, ny, nz), dtype=np.float32)
    ay_new = np.zeros((nx, ny, nz), dtype=np.float32)
    az_new = np.zeros((nx, ny, nz), dtype=np.float32)
    bx_new = np.zeros((nx, ny, nz), dtype=np.float32)
    by_new = np.zeros((nx, ny, nz), dtype=np.float32)
    bz_new = np.zeros((nx, ny, nz), dtype=np.float32)
    v0_data = [vf[0] for vf in images.timestamps]
    f0_data = [vf[1] for vf in images.timestamps]

    vd = v0_data - v0_data[0]
    fd = f0_data - f0_data[0]

    A = np.vstack((vd[1:], fd[1:])).T
    At = A.T
    AtA = At @ A
    d = np.zeros((nbin - 1, nz), dtype=np.float32)
    n_iter = 10
    for ix in tqdm(range(nx)):
        for iy in range(nx):
            for i in range(1, nbin):
                d[i - 1, :] = dvf_dct[i][0].data[ix, iy, :]
            b = At @ d
            x = np.zeros((2, nz), dtype=np.float32)
            for _ in range(n_iter):
                r0 = b - AtA @ x
                if np.sum(np.abs(r0) == 0) > 0:
                    break
                alpha = np.sum(r0 ** 2, axis=0) / \
                    np.sum((AtA @ r0) * r0, axis=0)
                x += alpha * r0
            ax_new[ix, iy, :] = x[0, :]
            bx_new[ix, iy, :] = x[1, :]

            for i in range(1, nbin):
                d[i - 1, :] = dvf_dct[i][1].data[ix, iy, :]
            b = At @ d
            x = np.zeros((2, nz), dtype=np.float32)
            for _ in range(n_iter):
                r0 = b - AtA @ x
                if np.sum(np.abs(r0) == 0) > 0:
                    break
                alpha = np.sum(r0 ** 2, axis=0) / \
                    np.sum((AtA @ r0) * r0, axis=0)
                x += alpha * r0
            ay_new[ix, iy, :] = x[0, :]
            by_new[ix, iy, :] = x[1, :]

            for i in range(1, nbin):
                d[i - 1, :] = dvf_dct[i][2].data[ix, iy, :]
            b = At @ d
            x = np.zeros((2, nz), dtype=np.float32)
            for _ in range(n_iter):
                r0 = b - AtA @ x
                if np.sum(np.abs(r0) == 0) > 0:
                    break
                alpha = np.sum(r0 ** 2, axis=0) / \
                    np.sum((AtA @ r0) * r0, axis=0)
                x += alpha * r0
            az_new[ix, iy, :] = x[0, :]
            bz_new[ix, iy, :] = x[1, :]
    return DeformParameter(ax_new, ay_new, az_new, bx_new, by_new, bz_new, center=img0.center, size=img0.size)
