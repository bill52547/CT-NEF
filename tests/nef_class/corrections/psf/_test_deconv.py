# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: _test_deconv.py
@date: 5/8/2019
@desc:
'''
# import nefct as nef
# from matplotlib import pyplot as plt
# %matplotlib inline
# 
# i = 40
# pnt_img_path = '/mnt/nfs/NefDatabase/data/long/new_data_1.4_psf_xy/img_xy_' + str(i) + '_psf.hdf5'
# image = nef.nef_classes.data.corrections.psf.PointSource.json_load(pnt_img_path)
# deconv = nef.nef_classes.functions.corrections.psf.Deconvolute(10, fitx, fity, fitz)
# deconv.make_kernel_xy(image)
# deconv.make_kernel_z(image)
# image1 = deconv(image)
# plt.figure(figsize = (15, 15))
# plt.subplot(211)
# image[120:160, 80:120, :].imshow()
# plt.colorbar()
# plt.subplot(212)
# image1[120:160, 80:120, :].imshow()
# plt.colorbar()
