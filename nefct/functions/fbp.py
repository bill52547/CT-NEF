from nefct import nef_class
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.project import Project
from nefct.functions.back_project import BackProject
import tensorflow as tf
from nefct.utils import tqdm
from scipy.fftpack import fft, ifft

'''ramp filter based fbp'''
@nef_class
class FBP:
    n_iter: int
    back_project: BackProject

    def __call__(self, projection: ProjectionSequence) -> Image:
        proj_data_ = projection.data
        for i in tqdm(range(proj_data_.shape[-1])):
            f = fft(proj_data_[:, i])
            f *= np.abs(projection.scanner.detector.meshgrid)
            proj_data_[:, i] = ifft(f)
        x = self.back_project(projection.update(data - proj_data_))
        return x
