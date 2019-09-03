# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_angled_normalization_correction.py
@date: 5/9/2019
@desc:
'''
# import numpy as np
# import h5py
# %matplotlib inline
# from matplotlib import pyplot as plt
# import nefct as nef
#
# block = nef.nef_classes.config.Block([20.0, 51.3, 51.3], [1, 15, 15])
# scanner = nef.nef_classes.config.PetCylindricalScanner(400.0, 420.0, 26, 48, 0.0, block)
#
# shape = [200, 200, 500]
# center = [0.,0.,0.]
# size = [400., 400., 500 * 3.42]
# # with h5py.File('/mnt/gluster/Techpi/brain16/recon/data/cylinder/small_cylinder_air_trans.h5', 'r') as fin:
# #     fst = np.array(fin['listmode_data']['fst'])
# #     snd = np.array(fin['listmode_data']['snd'])
# with h5py.File('/mnt/nfs/NefDatabase/data/long/data_1.4_regular/input_trans.h5', 'r') as fin:
#     fst = np.array(fin['listmode_data']['fst'])
#     snd = np.array(fin['listmode_data']['snd'])
# # Lx = fst[:, 0] - snd[:, 0]
# # Ly = fst[:, 1] - snd[:, 1]
# # Lz = fst[:, 2] - snd[:, 2]
#
# # L = np.sqrt(Lx ** 2 + Ly ** 2)
# # np.sum(L < np.abs(Lz))
#
# # lors = nef.nef_classes.functions.LorsFromFstSnd()(fst[L > np.abs(Lz),:], snd[L > np.abs(Lz),:])
# lors = nef.nef_classes.functions.LorsFromFstSnd()(fst, snd)
#
# listmode = nef.nef_classes.data.Listmode(np.ones((lors.length,), dtype = np.float32), lors)
#
# listmode_cos_ = nef.nef_classes.functions.corrections.normalization.NormalizationAngledCorrect()(listmode)
#
# image = nef.nef_classes.data.Image(np.zeros(shape, dtype = np.float32), center, size)
# emap = nef.nef_classes.functions.EmapGenerator('ring', scanner)(image)
# emap.data[emap.data < 1e-6] = 1e-6
# mlem = nef.nef_classes.functions.Mlem('tf-eager', 15, emap)
# img = mlem(listmode)
# img_cos = mlem(listmode_cos_)
