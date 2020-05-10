import attr
import numpy as np
from nefct.base.base import nef_class


@nef_class
class DeformParameter:
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray
    center: np.ndarray
    size: np.ndarray

    def __attrs_post_init__(self):
        if self.cx is None:
            object.__setattr__(self, 'cx', self.ax * 0)
            object.__setattr__(self, 'cy', self.ax * 0)
            object.__setattr__(self, 'cz', self.ax * 0)

    def make_full(self, v: float, f: float):
        return (self.ax * v + self.bx * f + self.cx,
                self.ay * v + self.by * f + self.cy,
                self.az * v + self.bz * f + self.cz)

    def update_para(self, ax, ay, az, bx, by, bz):
        object.__setattr__(self, 'ax', ax)
        object.__setattr__(self, 'ay', ay)
        object.__setattr__(self, 'az', az)
        object.__setattr__(self, 'bx', bx)
        object.__setattr__(self, 'by', by)
        object.__setattr__(self, 'bz', bz)
