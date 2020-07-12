from nefct import nef_class
import numpy as np
from nefct.data.image import Image
from nefct.data.projection import ProjectionSequence
from nefct.functions.project import Project
from nefct.geometry.scanner_config import ScannerConfig
from nefct.functions.back_project import Backproject
from nefct.utils import tqdm


@nef_class
class SART:
    n_iter: int
    lambda_: float
    scanner: ScannerConfig

    def __call__(self, projection: ProjectionSequence, x: Image = None) -> Image:

        angles = projection.angles
        offsets_a = projection.offsets_a
        offsets_b = projection.offsets_b
        x_out = x * 0
        projector = Project(self.scanner, angles, offsets_a, offsets_b)
        bprojector = Backproject(self.scanner)

        emap = bprojector(projector(x_out + 1), x)
        emap.data[emap.data == 0] = 1e8

        for iter in tqdm(range(self.n_iter)):
            proj = projector(x_out)
            bproj = bprojector(projection - proj, x)
            x_out = x_out + (bproj / emap) * self.lambda_
        return x_out
