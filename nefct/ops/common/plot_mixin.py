# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: imshow_mixin.py
@date: 5/6/2019
@desc:
'''


class PlotMixin:
    def __call__(self, *args, **kwargs):
        pass

    def plot(self, i: int = 0, *args, **kwargs) -> None:
        from matplotlib import pyplot as plt
        if len(self.shape) == 1:
            plt.plot(self.data, *args, **kwargs)
        elif len(self.shape) == 2:
            plt.plot(self.data[:, i], *args, **kwargs)
        else:
            raise ValueError
