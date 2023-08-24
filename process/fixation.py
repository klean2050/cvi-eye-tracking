import numpy as np, cv2, glob
from scipy import stats


def fix_bounds(data, new_res=False):
    data = [d for d in data if (d[0] != 0) or (d[1] != 0)]
    for i, (x, y) in enumerate(data):
        if new_res:
            # up left corner is (320, 240), total is (1920, 1200)
            x = 1919 if x >= 1920 else 0 if x < 0 else x
            y = 1199 if y >= 1200 else 0 if y < 0 else y
        else:
            x = 1279 if x >= 1280 else 0 if x < 0 else x
            y = 719 if y >= 720 else 0 if y < 0 else y
        data[i] = [int(x), int(y)]
    return np.array(data)


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
            fixation = fix_bounds(fixation, new_res=self.new_res)
            saliency = [smap[int(x), int(y)] for x, y in fixation]
            trace.extend(saliency)
        return trace

    def average_saliency(self, smap):
        trace = self.fixation_trace(smap)
        return np.mean(trace) if trace else 0

    def number_of_fixations(self):
        return len(self.fixations)

    def duration_of_fixations(self):
        durs = [f["duration"] for f in self.fixations]
        if durs:
            return np.mean(durs), np.std(durs)
        else:
            return 0, 0

    def latency_first_fixation(self):
        if len(self.fixations):
            this_fixation = self.fixations[0]
            return this_fixation["latency"]
        else:
            return 0

    def saliency_first_fixation(self, smap):
        if len(self.fixations):
            this_fixation = self.fixations[0]
            saliency = [smap[int(x), int(y)] for x, y in this_fixation["data"]]

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
            saliency = [smap[int(x), int(y)] for x, y in this_fixation["data"]]
            return np.mean(saliency)
        else:
            return 0

    def latency_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append((fixation["latency"], np.mean(saliency)))

        if len(saliencies):
            saliencies.sort(key=lambda x: x[1])
            return saliencies[-1][0]
        else:
            return 0

    def saliency_maxsal_fixation(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append(np.mean(saliency))

        if len(saliencies):
            saliencies.sort()
            return saliencies[-1]
        else:
            return 0
        
    def latency_overtime(self):
        latencies = [f["latency"] for f in self.fixations]
        t = np.arange(len(latencies))
        if len(latencies) > 1:
            return stats.linregress(t, latencies).slope
        else:
            return 0

    def saliency_overtime(self, smap):
        saliencies = []
        for fixation in self.fixations:
            saliency = [smap[int(x), int(y)] for x, y in fixation["data"]]
            saliencies.append(np.mean(saliency))

        t = np.arange(len(saliencies))
        if len(saliencies) > 1:
            return stats.linregress(t, saliencies).slope
        else:
            return 0