from .color import *
from .depth import *
from .object import *
from .angle import *


class SaliencyMap:
    def __init__(self, smap_type):
        self.smap_type = smap_type

    def get_smap(self, trial):
        if self.smap_type == "intensity":
            return smap_intensity(trial)
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
            return self.smap_object(trial)
        else:
            return NotImplementedError
