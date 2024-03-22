import numpy as np
from scipy.signal import medfilt
from utils import fix_bounds


class SaccadeAnalyzer:
    def __init__(self, root, saccades, keep_outliers=True):
        self.root = root
        self.saccades = saccades
        self.new_res = True if "new_res" in root else False

        # count saccades outside of the image
        self.out_saccades = 0
        for i, sac in enumerate(self.saccades):
            sac["data"] = fix_bounds(sac["data"], new_res=self.new_res)
            if np.count_nonzero(np.isnan(sac["data"])):
                self.out_saccades += 1

        # mark if there are no saccades in the image
        if not keep_outliers:
            self.saccades = [f for f in self.saccades if not np.isnan(f["data"]).any()]
        self.zero_saccades = True if len(self.saccades) == 0 else False

        # smooth eye tracks during saccades
        for i, sac in enumerate(self.saccades):
            self.saccades[i]["data"][:, 0] = medfilt(sac["data"][:, 0], kernel_size=5)
            self.saccades[i]["data"][:, 1] = medfilt(sac["data"][:, 1], kernel_size=5)

    def saccade_number(self):
        return len(self.saccades)

    def saccade_duration(self):
        durs = [f["duration"] for f in self.saccades]
        return np.mean(durs)

    def average_saliency(self, smap, when="start"):
        assert when in ["start", "end"], "<when> must be 'start' or 'end'"
        saliencies = []
        for saccade in self.saccades:
            data = saccade["data"]
            x, y = data[0] if when == "start" else data[-1]
            saliencies.append(smap[int(x), int(y)])
        return np.mean(saliencies)

    def saccade_velocity(self):
        velocities = []
        for saccade in self.saccades:
            velocity = np.diff(saccade["data"], axis=0)
            velocity = np.mean(velocity, axis=1)
            velocities.append(np.linalg.norm(velocity))
        return np.mean(velocities)

    def get_saliencies(self, smap):
        saliencies = []
        for saccade in self.saccades:
            saliency = [smap[int(x), int(y)] for x, y in saccade["data"]]
            saliencies.append(np.nanmean(saliency))
        return saliencies

    def get_durations(self):
        return [f["duration"] for f in self.saccades]
