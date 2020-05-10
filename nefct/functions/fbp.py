from nefct import nef_class
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.back_project import BackProject
from nefct.utils import tqdm
from nefct.data import Image2D
import astra

'''ramp filter based fbp'''


@nef_class
class FBP:
    back_project: BackProject

    def __call__(self, projection: ProjectionSequence) -> Image:
        vol_geom = astra.create_vol_geom(self.back_project.shape)
        unit_size_ = self.back_project.unit_size

        if projection.scanner.detector.offset == 0:
            proj_data = projection.data.transpose()[:, ::-1]
            det_num_plus = projection.scanner.detector.number
        elif projection.scanner.detector.offset > 0:
            proj_data_ = projection.data.transpose()
            det_num = (projection.scanner.detector.offset //
                       projection.scanner.detector.unit_size).astype(np.int32)
            off_data = np.zeros((len(projection.angles), det_num * 2))
            proj_data = np.concatenate((off_data, proj_data_), axis = 1)
            det_num_plus = projection.scanner.detector.number + det_num * 2
            proj_data[:, :det_num_plus // 2] = 0
            proj_data = proj_data[:, ::-1]
        else:
            raise NotImplementedError
        proj_geom = astra.create_proj_geom('fanflat',
                                           projection.scanner.detector.unit_size / unit_size_,
                                           det_num_plus,
                                           projection.angles,
                                           projection.scanner.SAD / unit_size_,
                                           (
                                                   projection.scanner.SID - projection.scanner.SAD) / unit_size_)
        sinogram_id = astra.data2d.create('-sino', proj_geom, proj_data)
        rec_id = astra.data2d.create('-vol', vol_geom)
        cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['option'] = {'FilterType': 'Ram-Lak'}

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec_img_data = astra.data2d.get(rec_id)

        return Image2D(rec_img_data, [0, 0], [unit_size_ * s for s in self.back_project.shape])
