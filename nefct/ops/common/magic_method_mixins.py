# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: magic_method_mixins.py
@date: 5/7/2019
@desc:
'''
__all__ = ('GetItemMixin',)


class GetItemMixin:
    def __getitem__(self, *args, **kwargs):
        return self.update(data = self.data.__getitem__(*args, **kwargs))
