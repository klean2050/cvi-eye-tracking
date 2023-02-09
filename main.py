import os, numpy as np
import matplotlib.pyplot as plt


def plot_fixation_stats(names, stats, title, outpath):
    t = list(range(len(names)))
    plt.figure(figsize=(10, 5))
    plt.bar(names, stats)
    plt.xticks(t, names, rotation=90)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.title(title)
    plt.savefig(outpath)


def plot_distance_matrix(names, dmatrix, title, outpath):
    t = list(range(len(names)))
    fig, ax = plt.figure(figsize=(10, 5))
    img = ax.imshow(dmatrix)
    ax.set_xticks(t)
    ax.set_xticklabels(names, rotation=90)
    fig.colorbar(img)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(outpath)


if __name__ == "__main__":
    root = "/home/kavra/Datasets/medical/cvi_eyetracking/asc_data_v1/"

    ids = [i for i in os.listdir(root) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.split("_")[0].startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i not in ctrl_ids]
    names = ctrl_ids + cvi_ids

    trials_images_subset = [
        "Freeviewingstillimage_1.jpg",
        "Freeviewingstillimage_2.jpg",
        "Freeviewingstillimage_4.jpg",
        "Freeviewingstillimage_5.jpg",
        "Freeviewingstillimage_7.jpg",
        "Freeviewingstillimage_8.jpg",
        "Freeviewingstillimage_9.jpg",
        "Freeviewingstillimage_10.jpg",
        "Freeviewingstillimage_10_cutout.tif",
        "Moviestillimage_8.jpg",
        "Moviestillimage_6.jpg",
        "Freeviewingstillimage_50.jpg",
        "Freeviewingstillimage_88_cutout.tif",
    ]
    trial = "Freeviewingstillimage_1.jpg"

    """
    def read_all_subjects(self, root, trial_name, vel=False):
        subject_ids = glob.glob(os.path.join(root, "*.asc"))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]

        self.timeseries = {}
        self.data_frac = {}
        for subject in subject_ids:
            sub = Subject(subject)
            trial_data, frac = sub.extract_data(trial_name, vel)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - frac
            
    self.read_all_subjects(trial_name, vel)
    subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
    subject_ids.sort()

    distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
    for i, keyi in enumerate(subject_ids):
        for j, keyj in enumerate(subject_ids):
            distance_matrix[i, j] = dtw(
                self.timeseries[keyi][:1000], self.timeseries[keyj][:1000]
             )
    """
