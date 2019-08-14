# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: imshow_mixin.py
@date: 5/6/2019
@desc:
'''


class ImshowMixin:
    def __call__(self, *args, **kwargs):
        pass

    def imshow(self, *args, **kwargs) -> None:
        from matplotlib import pyplot as plt
        if self.shape == 2:
            plt.imshow(self.data, *args, **kwargs)
        else:
            plt.imshow(self.data[:, :, int(1 + self.shape[2] / 2)], *args, **kwargs)

    def imshow3d(self, *args, **kwargs) -> None:
        from matplotlib import pyplot as plt
        plt.figure(figsize = (15, 5))
        plt.subplot(131)
        plt.imshow(self.central_slices[0], *args, **kwargs)
        plt.subplot(132)
        plt.imshow(self.central_slices[1], *args, **kwargs)
        plt.subplot(133)
        plt.imshow(self.central_slices[2], *args, **kwargs)
