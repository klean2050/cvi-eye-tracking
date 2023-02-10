import os, glob, cv2
from subject import Subject


class SaliencyTrace:
    def __init__(self, root, trial_name, smap_dir, maps=["all"]):
        self.root = root
        self.trial_name = trial_name
        self.ids = glob.glob(os.path.join(self.root, "*.asc"))
        self.ids = [os.path.basename(d)[:-4] for d in self.ids]
        self.smap_dir = smap_dir  # "cvi-extra/saliency_maps/"
        if "all" in maps:
            self.smaps = [
                "0",
                "90",
                "by",
                "color",
                "edges",
                "intensity",
                "object",
                "orientation",
                "rg",
                "grad",
            ]
        else:
            self.smaps = maps

    def load_trial_img(self):
        img = cv2.imread(os.path.join(self.root, self.trial_name))
        return img

    def load_saliency_map(self, smap):
        path = os.path.join(self.root, self.smap_dir, smap)
        return cv2.imread(path)

    def read_subjects(self, names, vel=False):
        data, frac = {}, {}
        for subject in names:
            sub = Subject(subject)
            trial_data, frac = sub.extract_data(self.trial_name, vel)
            data[subject] = trial_data
            frac[subject] = 1 - frac
        return data, frac

    def read_fixations(self, names):
        fixations = {}
        for subject in names:
            sub = Subject(subject)
            this = sub.extract_fixations(self.trial_name)
            fixations[subject] = this
        return fixations

    def extract_traces(self, names, smap):
        traces = {}
        for subject in names:
            sub = Subject(subject)
            this = sub.extract_trace(self.trial_name)
            traces[subject] = this
        return traces
