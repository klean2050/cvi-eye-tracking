class SaliencyMap:
    def __init__(self, smap_type):
        self.smap_type = smap_type

    def get_smap(self, trial):
        if self.smap_type == "intensity":
            return self.get_intensity_smap(trial)
        elif self.smap_type == "color":
            return self.get_color_smap(trial)
        elif self.smap_type == "rg":
            return self.get_rg_smap(trial)
        elif self.smap_type == "by":
            return self.get_by_smap(trial)
        elif self.smap_type == "orientation":
            return self.get_orientation_smap(trial)
        elif self.smap_type == "0":
            return self.get_0_smap(trial)
        elif self.smap_type == "90":
            return self.get_90_smap(trial)
        elif self.smap_type == "edges":
            return self.get_edges_smap(trial)
        elif self.smap_type == "grad":
            return self.get_grad_smap(trial)
        elif self.smap_type == "depth":
            return self.get_depth_smap(trial)
        elif self.smap_type == "object":
            return self.get_object_smap(trial)
        else:
            return NotImplementedError

    def get_intensity_smap(self, trial):
        return trial.image
