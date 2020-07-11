from nefct.base.base import nef_class

@nef_class
class Timestamps:
    data: list

    def v_data(self):
        return [vf[0] for vf in self.data]

    def f_data(self):
        return [vf[1] for vf in self.data]