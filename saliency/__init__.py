from .color import *
from .depth import *
from .object import *
from .angle import *
from .basic import *
from .deepg import *


class SaliencyMap:
    def __init__(self, smap_type):
        self.smap_type = smap_type

    def get_smap(self, trial):
        if self.smap_type == "intensity":
            return smap_intensity(trial)
        elif self.smap_type == "rough":
            return smap_rough(trial)
        elif self.smap_type == "fine":
            return smap_fine(trial)
        elif self.smap_type == "color":
            return smap_color(trial)
        elif self.smap_type == "rg":
            return smap_rg_opponency(trial)
        elif self.smap_type == "by":
            return smap_by_opponency(trial)
        elif self.smap_type == "orientation":
            return self.smap_orientation(trial)
        elif isinstance(self.smap_type, int):
            return smap_angle(trial, self.smap_type)
        elif self.smap_type == "depth":
            return smap_depth(trial)
        elif isinstance(self.smap_type, tuple):
            return smap_object(trial)
        elif self.smap_type == "deepgaze":
            return smap_deepgaze(trial)
        elif self.smap_type == "center":
            return smap_center(trial)
        else:
            return NotImplementedError
