# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: conj_grad.py
@date: 8/28/2019
@desc:
'''
from nefct import nef_class, Any
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.data.total_variation import TotalVariation
from nefct.functions.project import Project
from nefct.functions.back_project import BackProject
from nefct.functions.total_variation import TotalVari, InvertTotalVari, TotalVariT
from nefct.utils import tqdm
from copy import deepcopy

''' to solve
x = argmin norm(Ax - y, 2) / 2 + norm(Wx - d, 2) * mu / 2
d = z - u
'''


@nef_class
class ConjGrad:
    n_iter: int
    project: Project
    back_project: BackProject
    total_vari: TotalVari
    invert_total_vari: InvertTotalVari
    mu: float

    def __call__(self, projection: ProjectionSequence, d: TotalVariation = None) -> Image:
        x = Image(np.zeros(self.back_project.shape),
                  [0] * len(self.back_project.shape),
                  [self.back_project.unit_size * s for s in self.back_project.shape])
        if self.total_vari is None:
            b = self.back_project(projection)
            A = lambda x: self.back_project(self.project(x))

        else:
            b = self.back_project(projection) + self.invert_total_vari(d) * self.mu
            A = lambda x: self.back_project(self.project(x)) + \
                          self.invert_total_vari(self.total_vari(x)) * self.mu
        r = b - A(x)
        p = deepcopy(r)
        rsold = np.sum((r.data ** 2).data)

        for _ in range(self.n_iter):
            Ap = A(p)
            alpha = rsold / np.sum(p.data * Ap.data)
            x = x + p * alpha
            r = r - Ap * alpha
            rsnew = np.sum(r.data ** 2)
            p = r + p * (rsnew / rsold)
            rsold = rsnew

        return x
