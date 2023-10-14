import os
from scipy import stats
from process import *
from utils import *


TRIALS = {
    "all": TRIAL_LIST,
    "texture": TRIAL_TEXTURE,
    "complexity": TRIAL_COMPLEXITY,
    "orientation": TRIAL_ORIENTATION,
    "brightness": TRIAL_BRIGHTNESS,
    "contrast": TRIAL_CONTRAST,
    "color": TRIAL_COLOR,
    "depth": TRIAL_DEPTH,
    "face": TRIAL_FACE,
    "movement": TRIAL_MOVEMENT,
    "face_texture": TRIAL_FACE_TEXTURE,
}
these_trials = TRIALS["face"]
smap = "face"


def statistic_mw(x, y):
    return stats.mannwhitneyu(x, y)[0]


def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1


def extract_features(ids, trial, smap):
    if smap is not None:
        smap = trial.load_saliency_map(smap)
        smap_vs = trial.load_saliency_map("red")

    features = []
    for subject in ids:
        sub = Subject(DATA_ROOT, subject)
        out = sub.extract_fixations(trial_name=this_trial.trial_name)
        analyzer = FixationAnalyzer(DATA_ROOT, out)
        feat = analyzer.average_saliency(smap)
        feat_vs = analyzer.average_saliency(smap_vs)
        # feat = analyzer.number_of_fixations()
        # feat = sub.eye_mov_entropy(trial.trial_name, perplexity=True)
        features.append(feat - feat_vs)
        feats[subject].append(feat - feat_vs)
    return features


if __name__ == "__main__":
    ids = [i for i in os.listdir(DATA_ROOT) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i.startswith("1")]

    feats = {name.split(".")[0]: [] for name in ids}

    p_values = []
    for trial in these_trials:
        this_trial = ImageTrial(DATA_ROOT, trial, "smaps")

        features_ctrl = extract_features(ctrl_ids, this_trial, smap)
        features_cvi = extract_features(cvi_ids, this_trial, smap)

        stat, p_value = stats.mannwhitneyu(features_ctrl, features_cvi)
        significance = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        p_values.append(p_value)
        print(trial, "\t", np.round(p_value, 4), significance)

    bonferroni = np.round(0.05 / len(these_trials), 4)
    combined = np.round(stats.combine_pvalues(p_values)[1], 6)
    print("\nSignificance requirement: p <", bonferroni)
    print("Combined (Fisher's method) p =", combined)

    feats = {k: np.mean(v) for k, v in feats.items()}
    feats_ctrl = [v for k, v in feats.items() if k in ctrl_ids]
    feats_cvi = [v for k, v in feats.items() if k in cvi_ids]

    print("\nMedian ~ Control:", np.median(feats_ctrl), "CVI:", np.median(feats_cvi))
    print("IQR ~ Control:", iqr(feats_ctrl), "CVI:", iqr(feats_cvi))

    cohens_d = (np.mean(feats_ctrl) - np.mean(feats_cvi)) / (
        np.sqrt(
            (
                (len(feats_ctrl) - 1) * np.std(feats_ctrl) ** 2
                + (len(feats_cvi) - 1) * np.std(feats_cvi) ** 2
            )
            / (len(feats_ctrl) + len(feats_cvi) - 2)
        )
    )
    print("Cohen's d =", cohens_d)
    print(stats.mannwhitneyu(feats_ctrl, feats_cvi))
    permutation = stats.permutation_test((feats_ctrl, feats_cvi), statistic_mw)
    print("Permutation test p =", permutation.pvalue)
