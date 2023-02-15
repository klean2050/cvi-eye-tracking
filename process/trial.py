import os, glob, cv2
from process import Subject
from saliency import SaliencyMap
from skimage import measure


class ImageTrial:
    def __init__(self, root, trial_name, smap_dir):
        self.root = root
        self.path = glob.glob(f"trials/*/*{trial_name}")[0]
        self.trial_name = trial_name
        self.smap_dir = smap_dir
        self.ids = glob.glob(os.path.join(root, "*.asc"))
        self.ids = [os.path.basename(d)[:-4] for d in self.ids]

    def load_trial_img(self):
        img = cv2.imread(self.path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_saliency_map(self, smap_type):
        filename = f"{self.trial_name[:-4]}_{smap_type}.jpg"
        path = os.path.join(self.smap_dir, filename)
        if os.path.exists(path):
            return cv2.imread(path)
        else:
            sal = SaliencyMap(smap_type)
            smap = sal.get_smap(self.load_trial_img())
            cv2.imwrite(path, smap * 255)
            return smap

    def complexity(self):
        img = self.load_trial_img()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        stats = measure.regionprops(img)
        areas = [l.area for l in stats]
        rp_tot = img.shape[0] * img.shape[1]
        return sum(areas > (rp_tot / 25000))

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
