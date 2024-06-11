import os, seaborn as sns, json
from scipy import stats
from tqdm import tqdm

from process import *
from utils import *
from stat_helper import *


these_trials = TRIAL_LIST
smap_type = "face"
smap_type_vs = None
cutouts = False


def plot_number_of_fixations(saliencies_ctrl, saliencies_cvi, reverse=False):
    if reverse:
        saliencies_ctrl = 255 - np.array(saliencies_ctrl)
        saliencies_cvi = 255 - np.array(saliencies_cvi)

    plt.figure(figsize=(9, 3), dpi=200)
    sns.kdeplot(
        saliencies_ctrl,
        color="#73afef",
        linewidth=0.25,
        fill=True,
        alpha=0.3,
        label="CTRL",
    )
    sns.kdeplot(
        saliencies_cvi,
        color="limegreen",
        linewidth=0.25,
        fill=True,
        alpha=0.3,
        label="CVI",
    )
    plt.legend()
    sns.despine()
    # remove y-axis numbers
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.xlabel("Fixation Saliency")
    plt.ylabel("Fixation Density")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"output/hist.png")
    plt.show()


def integrate_subjects(feats):
    new_feats = {}
    for subject in feats:
        real_subject = subject.split("_")[0]
        if real_subject not in new_feats:
            new_feats[real_subject] = []
        new_feats[real_subject].extend(feats[subject])
    return new_feats


def extract_features(ids, trial):
    # construct saliency map
    if smap_type is not None:
        smap = trial.load_saliency_map(smap_type)
        smap = smap.astype(np.float32)
    if smap_type_vs is not None:
        smap_vs = trial.load_saliency_map(smap_type_vs)
        smap_vs = smap_vs.astype(np.float32)
        smap -= smap_vs
    # normalize saliency map
    smap *= 255 / np.max(abs(smap))

    new_res = False
    for subject in ids:
        # load data and extract fixations
        sub = Subject(which_root[subject], subject)
        fixs = sub.extract_fixations(trial_name=this_trial.trial_name)
        analyzer = FixationAnalyzer(which_root[subject], fixs)
        if analyzer.zero_fixations:
            continue

        # adjust for the change in resolution
        if "new_res" in which_root[subject] and not new_res:
            new_res = True
            if smap_type is not None:
                smap = np.pad(smap, ((320, 320), (240, 240)), "constant")
            if smap_type_vs is not None:
                smap_vs = np.pad(smap_vs, ((320, 320), (240, 240)), "constant")

        # compute and store feature
        feat = analyzer.average_saliency(smap)
        if feat is not None:
            feats[subject].append(feat)
        saliencies[subject][trial.trial_name] = analyzer.get_saliencies(smap)


if __name__ == "__main__":
    # get list of subject data
    ids1 = [i for i in os.listdir(DATA_ROOT1) if i.endswith(".asc")]
    ids2 = ids1 + [i for i in os.listdir(DATA_ROOT2) if i.endswith(".asc")]
    ids = ids2 + [i for i in os.listdir(DATA_ROOT3) if i.endswith(".asc")]

    # initialize features and paths
    feats = {name.split(".")[0]: [] for name in ids}
    saliencies = {name.split(".")[0]: {} for name in ids}
    which_root = {
        name.split(".")[0]: (
            DATA_ROOT1 if i < len(ids1) else DATA_ROOT2 if i < len(ids2) else DATA_ROOT3
        )
        for i, name in enumerate(ids)
    }

    # distinguish controls from CVI subjects
    ctrl_ids = [i.split(".")[0] for i in ids if i.startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i.startswith("1")]
    ctrl_subjects = set([i.split("_")[0] for i in ctrl_ids])
    cvi_subjects = set([i.split("_")[0] for i in cvi_ids])

    p_values = []
    for trial in tqdm(these_trials):
        if "cutout" in trial and not cutouts:
            continue
        # extract features for this trial
        this_trial = ImageTrial(trial, "smaps")
        extract_features(ctrl_ids, this_trial)
        extract_features(cvi_ids, this_trial)

    # save saliences in json
    with open(f"saliencies_{smap_type}.json", "w") as f:
        json.dump(saliencies, f, indent=4)

    """# aggregate features per subject (only if more than one trial)
    feats = integrate_subjects(feats)
    feats = {k: np.mean(v) for k, v in feats.items() if len(v) > 1}
    saliencies = integrate_subjects(saliencies)
    saliencies = {k: v for k, v in saliencies.items() if len(v) > 1}

    # save features for plotting
    feats_ctrl = [v for k, v in feats.items() if k.startswith("2")]
    feats_cvi = [v for k, v in feats.items() if k.startswith("1")]
    np.save(f"feats_ctrl_{smap_type}.npy", np.array(feats_ctrl))
    np.save(f"feats_cvi_{smap_type}.npy", np.array(feats_cvi))
    # save the keys
    np.save(f"keys_ctrl.npy", np.array([k for k in feats.keys() if k.startswith("2")]))
    np.save(f"keys_cvi.npy", np.array([k for k in feats.keys() if k.startswith("1")]))

    # same for saliencies
    saliencies_ctrl = [v for k, v in saliencies.items() if k.startswith("2")]
    saliencies_ctrl = [v for sublist in saliencies_ctrl for v in sublist]
    saliencies_cvi = [v for k, v in saliencies.items() if k.startswith("1")]
    saliencies_cvi = [v for sublist in saliencies_cvi for v in sublist]
    plot_number_of_fixations(saliencies_ctrl, saliencies_cvi)

    # print summary of aggregate statistics
    print("Mean ~ Control:", np.mean(feats_ctrl), "CVI:", np.mean(feats_cvi))
    print("Std ~ Control:", np.std(feats_ctrl), "CVI:", np.std(feats_cvi))
    print("Cohen's d =", cohen_d(feats_ctrl, feats_cvi))
    print(stats.mannwhitneyu(feats_ctrl, feats_cvi))
    print(stats.permutation_test((feats_ctrl, feats_cvi), statistic_mw))"""
