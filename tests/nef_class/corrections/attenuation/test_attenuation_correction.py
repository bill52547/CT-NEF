# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: test_attenuation_correction.py
@date: 5/9/2019
@desc:
'''
# import numpy as np
# import h5py
# %matplotlib inline
#
# import nefct as nef
# import matplotlib.pyplot as plt
#
# block = nef.nef_classes.config.Block([20, 53.3, 53.3], [1, 13, 13])
# scanner = nef.nef_classes.config.PetCylindricalScanner(424.5, 444.5, 4, 48, 0, block)
# shape = [100, 100, 26]
# center = [0.,0.,0.]
# size = [200., 200., 213.2]
#
# with h5py.File('/mnt/nfs/NefDatabase/data/mct/jaszczak/attenuation/input_trans.h5', 'r') as fin:
#     fst = np.array(fin['listmode_data']['fst'])
#     snd = np.array(fin['listmode_data']['snd'])
#
# lors = nef.nef_classes.functions.LorsFromFstSnd()(fst, snd)
#
# listmode = nef.nef_classes.data.Listmode(np.ones((lors.length,), dtype = np.float32), lors)
#
# listmode_cos_ = nef.nef_classes.functions.corrections.normalization.NormalizationAngledCorrect()(listmode)
#
# image = nef.nef_classes.data.Image(np.zeros(shape, dtype = np.float32), center, size)
# emap = nef.nef_classes.functions.EmapGenerator('ring', scanner)(image)
# emap = emap + 1e-8
# mlem = nef.nef_classes.functions.Mlem('tf-eager', 15, emap)
# img = mlem(listmode)
#
# with h5py.File('/mnt/nfs/NefDatabase/data/mct/jaszczak/attenuation/u_map.hdf5', 'r') as fin:
#     umap_data = np.array(fin['data'])
#     umap_center = np.array(fin['center']).tolist()
#     umap_size = np.array(fin['size']).tolist()
# umap = nef.nef_classes.data.Umap(umap_data, umap_center, umap_size)
#
# atten_corr = nef.nef_classes.functions.corrections.attenuation.AttenuationCorrect()
# atten_corr.make_u_map_listmode(umap, lors)
#
# listmode_corr = atten_corr(listmode)
# img_corr = mlem(listmode_corr)
# img_corr.imshow()
