# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: image.py
@date: 3/20/2019
@desc:
'''

from nefct import nef_class, List, Any
from nefct.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from nefct.ops.common.imshow_mixin import ImshowMixin
from nefct.ops.common.arithmetic_mixins import ArithmeticMixin
from nefct.ops.common.magic_method_mixins import GetItemMixin
from nefct.io import LoadMixin, SaveMixin


@nef_class
class ImageFourd(ShapePropertyMixin,
                 UnitSizePropertyMixin,
                 GetItemMixin,
                 CentralSlicesPropertyMixin,
                 CentralProfilesPropertyMixin,
                 LoadMixin,
                 SaveMixin,
                 ImshowMixin,
                 ArithmeticMixin):
    """
    Image data with center and size info.
    """

    data: Any
    center: List(float, 3)
    size: List(float, 3)
    timestamps: List(float)
    volumes: List(float)

    def __post_attr_init__(self):
        _n_view = self.data.shape[3]
        if self.timestamps is None:
            _time_stamp = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'timestamps', _time_stamp)

        if self.volumes is None:
            _volumes = [0.0 for _ in range(_n_view)]
            object.__setattr__(self, 'volumes', _volumes)

    def get_image_frame(self, ind: int):
        from .image import Image
        return Image(self.data[:, :, :, ind], self.center, self.size)
