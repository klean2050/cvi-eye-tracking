import os, glob, cv2
from process import Subject
from saliency import SaliencyMap


class ImageTrial:
    def __init__(self, root, trial_name, smap_dir):
        self.root = root
        self.trial_name = trial_name
        self.smap_dir = smap_dir
        self.ids = glob.glob(os.path.join(self.root, "*.asc"))
        self.ids = [os.path.basename(d)[:-4] for d in self.ids]

    def load_trial_img(self):
        img = cv2.imread(os.path.join(self.root, self.trial_name))
        return img

    def load_saliency_map(self, smap_type):
        filename = f"{self.trial_name[:-4]}_{smap_type}.png"
        path = os.path.join(self.root, self.smap_dir, filename)
        if os.exists(path):
            return cv2.imread(path)
        else:
            sal = SaliencyMap(smap_type)
            smap = sal.get_smap(self.load_trial_img())
            cv2.imwrite(path, smap)
            return smap

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
