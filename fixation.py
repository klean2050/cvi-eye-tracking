import numpy as np, cv2, glob
from scipy import stats
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
        self.fixations = fixations

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
        for fixation in data:
            fixation = fix_bounds(fixation)
            saliency = [smap[x, y] for x, y in fixation]
            trace.extend(saliency)
        return trace

    def average_saliency(self, smap):
        return np.mean(self.fixation_trace(smap))

    def number_of_fixations(self):
        return len(self.fixations)

    def duration_of_fixations(self):
        return sum([f["duration"] for f in self.fixations])

    def latency_first_fixation(self):
        if len(self.fixations):
            this_fixation = self.fixations[0]
            return this_fixation["latency"]
        else:
            return 0

    def saliency_first_fixation(self, smap):
        if len(self.fixations):
            this_fixation = self.fixations[0]
            saliency = [smap[x, y] for x, y in this_fixation["data"]]
            return np.mean(saliency)
        else:
            return 0

    def latency_longest_fixation(self):
        if len(self.fixations):
            self.fixations.sort(key=lambda x: x["duration"])
            this_fixation = self.fixations[-1]
            return this_fixation["latency"]
        else:
            return 0

    def saliency_longest_fixation(self, smap):
        if len(self.fixations):
            self.fixations.sort(key=lambda x: x["duration"])
            this_fixation = self.fixations[-1]
            saliency = [smap[x, y] for x, y in this_fixation["data"]]
            return np.mean(saliency)
        else:
            return 0

    def latency_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[x, y] for x, y in fixation["data"]]
            saliencies.append((fixation["latency"], np.mean(saliency)))

        if len(saliencies):
            saliencies.sort(key=lambda x: x[1])
            return saliencies[-1][0]
        else:
            return 0

    def saliency_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[x, y] for x, y in fixation["data"]]
            saliencies.append(np.mean(saliency))

        if len(saliencies):
            saliencies.sort()
            return saliencies[-1]
        else:
            return 0
