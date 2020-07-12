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
import numpy as np
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
    lamb_: float

    def __call__(self, projection: ProjectionSequence, x: Image = None, d: TotalVariation = None) -> Image:
        if x is None:
            x = self.back_project(projection) * 0

        if self.total_vari is None:
            b = self.back_project(projection)
            A = lambda x: self.back_project(self.project(x))

        else:
            b = self.back_project(projection) + self.invert_total_vari(d) * self.lamb_
            A = lambda x: self.back_project(self.project(x)) + \
                          self.invert_total_vari(self.total_vari(x)) * self.lamb_
        r = b - A(x)
        p = deepcopy(r)
        rsold = np.sum((r.data ** 2).data)
        r_past2 = 2 * 10 ** 100
        r_past1 = 1 * 10 ** 100
        r_curr = 1 * 10 ** 99
        tbar = tqdm(range(self.n_iter))
        for iter_ in tbar:
            Ap = A(p)
            if np.sum(Ap.data ** 2) < 100:
                break
            r_past2, r_past1 = r_past1, r_curr
            r_curr = np.sqrt(np.sum(r.data ** 2))

            if iter_ > 10 and r_past2 < r_past1 < r_curr:
#                 print('break at {:d} with {:.2e} {:.2e} {:.2e}'.format(iter_, r_past2, r_past1, r_curr))
                break
            # print(f'{iter_} / {self.n_iter}', ap_past2, ap_past1, ap_curr)
            alpha = rsold / np.sum(p.data * Ap.data)

            x = x + p * alpha
            r = r - Ap * alpha
            rsnew = np.sum(r.data ** 2)
            p = r + p * (rsnew / rsold)
            rsold = rsnew
        return x
