import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from nefct.base.base import nef_class
from nefct.data.deform_para import DeformParameter
from nefct.data.image import Image3D
from nefct.config import TF_USER_OP_PATH

deform_mod = tf.load_op_library(
    TF_USER_OP_PATH + '/tf_deform_tex_module.so'
)
deform_op = deform_mod.deform
# deform_op = deform_mod.deform
deform_invert_op = deform_mod.deform_invert

@nef_class
class Deform:
    dvf: DeformParameter

    def __call__(self, image: Image3D, v: float, f: float):
        mx = self.dvf.ax * v + self.dvf.bx * f + self.dvf.cx
        my = self.dvf.ay * v + self.dvf.by * f + self.dvf.cy
        mz = self.dvf.az * v + self.dvf.bz * f + self.dvf.cz

        nx, ny, nz = image.shape
        image_out = deform_op(image.data.transpose(),
                         mx.transpose(),
                         my.transpose(),
                         mz.transpose(),
                         nx = nx,
                         ny = ny,
                         nz = nz).numpy().transpose()
        return image.update(data = image_out)



@nef_class
class InvertDeform:
    dvf: DeformParameter

    def __call__(self, image: Image3D, v: float, f: float):
        mx = self.dvf.ax * v + self.dvf.bx * f + self.dvf.cx
        my = self.dvf.ay * v + self.dvf.by * f + self.dvf.cy
        mz = self.dvf.az * v + self.dvf.bz * f + self.dvf.cz

        nx, ny, nz = image.shape
        image_out = deform_invert_op(image.data.transpose(),
                         mx.transpose(),
                         my.transpose(),
                         mz.transpose(),
                         nx = nx,
                         ny = ny,
                         nz = nz).numpy().transpose()
        return image.update(data = image_out)