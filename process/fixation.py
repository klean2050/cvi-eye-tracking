import numpy as np, cv2, glob
from scipy import stats
from scipy.signal import medfilt

from utils import fix_bounds


def gkern(kernlen=21, nsig=3):
    """
    Returns a 2D Gaussian kernel
    """
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


class FixationAnalyzer:
    def __init__(self, root, fixations):
        self.root = root
        self.new_res = True if "new_res" in root else False

        # discard fixations of less than 50 ms
        self.fixations = [f for f in fixations if len(f["data"]) > 25]
        # discard fixations outside of the image
        for i, fix in enumerate(self.fixations):
            fix["data"] = fix_bounds(fix["data"], new_res=self.new_res)
            nan_count = np.count_nonzero(np.isnan(fix["data"]))
            # discard if more than 20% is out of image
            if nan_count > 0.2 * len(fix["data"]):
                self.fixations[i] = None
            else:
                self.fixations[i]["data"] = self.interpolate_fixation(fix["data"])

        # mark if there are no valid fixations
        self.fixations = [f for f in self.fixations if f is not None]
        self.zero_fixations = True if len(self.fixations) == 0 else False
        # smooth eye tracks during fixations
        for i, fix in enumerate(self.fixations):
            self.fixations[i]["data"][:, 0] = medfilt(fix["data"][:, 0], kernel_size=7)
            self.fixations[i]["data"][:, 1] = medfilt(fix["data"][:, 1], kernel_size=7)

    def interpolate_fixation(self, data):
        x, y = data.T
        x = np.ma.masked_invalid(x)
        y = np.ma.masked_invalid(y)
        x = np.ma.compressed(x)
        y = np.ma.compressed(y)
        xnew = np.arange(len(data))
        return np.array(
            [
                np.interp(xnew, np.arange(len(x)), x),
                np.interp(xnew, np.arange(len(y)), y),
            ]
        ).T

    def get_latencies(self):
        return [f["latency"] for f in self.fixations]

    def get_durations(self):
        return [f["duration"] for f in self.fixations]

    def get_stabilities(self):
        stabilities = []
        for fixation in self.fixations:
            x, y = np.array(fixation["data"], dtype=np.float32).T
            dx, dy = np.diff(x[:-1]), np.diff(y[:-1])
            stabilities.append(np.mean(np.sqrt(dx**2 + dy**2)))
        return stabilities

    def get_saliencies(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            fixation[f"saliency"] = np.nanmean(saliency)
            saliencies.append(np.nanmean(saliency))
        return saliencies

    def fixation_map(self, trial):
        path = glob.glob(f"trials/*/*{trial}")
        img = cv2.imread(path[0])
        out = np.zeros(img.shape[:-1])

        k = 101
        gaussian_filter = gkern(k, 2)
        for fixation in self.fixations:
            for i in fixation["data"]:
                x, y = np.array(i, dtype=np.float32)
                x, y = int(x), int(y)

                dx = int(k / 2)
                l = x - dx
                r = x + dx
                u = y - dx
                d = y + dx
                fl, fr = 0, k
                fu, fd = 0, k

                if l < 0:
                    fl = np.abs(l)
                    l = 0
                if r >= out.shape[1]:
                    fr = (k - 1) * (r - out.shape[1] - 1)
                    r = out.shape[1] - 1
                if u < 0:
                    fu = np.abs(u)
                    u = 0
                if d >= out.shape[0]:
                    fd = (k - 1) * (d - out.shape[0] - 1)
                    d = out.shape[0] - 1

                if (
                    out[u : d + 1, l : r + 1].shape
                    != gaussian_filter[fu : fd + 1, fl : fr + 1].shape
                ):
                    continue
                out[u : d + 1, l : r + 1] += gaussian_filter[fu : fd + 1, fl : fr + 1]

        return out

    def fixation_trace(self, smap):
        trace = []
        data = [f["data"] for f in self.fixations]
        for fix in data:
            saliency = [smap[int(x), int(y)] for x, y in fix]
            trace.extend(saliency)
        return trace

    def average_saliency(self, smap):
        return np.mean(self.fixation_trace(smap))

    def number_of_fixations(self):
        return len(self.fixations)

    def ratio_of_fixations(self, smap, thres=64):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append(np.nanmean(saliency))
        return len([s for s in saliencies if s > thres]) / len(saliencies)

    def average_duration(self):
        durs = [f["duration"] for f in self.fixations]
        return np.mean(durs)

    def average_stability(self):
        stabilities = []
        for fixation in self.fixations:
            x, y = np.array(fixation["data"], dtype=np.float32).T
            dx, dy = np.diff(x[:-1]), np.diff(y[:-1])
            stabilities.append(np.mean(np.sqrt(dx**2 + dy**2)))
        return np.nanmean(stabilities)

    def latency_first_fixation(self):
        return self.fixations[0]["latency"]

    def saliency_first_fixation(self, smap):
        this_fix = self.fixations[0]
        saliency = [smap[int(x), int(y)] for x, y in this_fix["data"]]
        return np.mean(saliency)

    def latency_longest_fixation(self):
        self.fixations.sort(key=lambda x: x["duration"])
        return self.fixations[-1]["latency"]

    def saliency_longest_fixation(self, smap):
        self.fixations.sort(key=lambda x: x["duration"])
        this_fix = self.fixations[-1]
        saliency = [smap[int(x), int(y)] for x, y in this_fix["data"]]
        return np.mean(saliency)

    def latency_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append((fixation["latency"], np.mean(saliency)))
        saliencies.sort(key=lambda x: x[1])
        return saliencies[-1][0]

    def saliency_maxsal_fixation(self, smap, norm=False):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append(np.mean(saliency))
        saliencies.sort()
        return saliencies[-1] / np.mean(smap) if norm else saliencies[-1]

    def duration_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append((fixation["duration"], np.mean(saliency)))
        saliencies.sort(key=lambda x: x[1])
        # get half of the fixations
        saliencies = saliencies[len(saliencies) // 2 :]
        return np.mean([s[0] for s in saliencies])
        # return saliencies[-1][0]

    def stability_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            x, y = np.array(fixation["data"], dtype=np.float32).T
            dx, dy = np.diff(x[:-1]), np.diff(y[:-1])
            stability = np.sqrt(dx**2 + dy**2)
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            if len(stability) > 1:
                saliencies.append((np.nanmean(stability), np.nanmean(saliency)))

        saliencies.sort(key=lambda x: x[1])
        return saliencies[-1][0]

    def latency_overtime(self):
        latencies = [f["latency"] for f in self.fixations]
        t = np.arange(len(latencies))
        if len(latencies) > 1:
            return stats.linregress(t, latencies).slope
        else:
            return 1e-5

    def saliency_overtime(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append(np.nanmean(saliency))

        t = np.arange(len(saliencies))
        if len(saliencies) > 1:
            return stats.linregress(t, saliencies).slope
        else:
            return 1e-5
