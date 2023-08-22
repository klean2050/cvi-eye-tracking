import os
from scipy import stats
from process import *
from utils import DATA_ROOT, TRIAL_LIST

if __name__ == "__main__":

    new_res = True if "new_res" in DATA_ROOT else False
    ids = [i for i in os.listdir(DATA_ROOT) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.split("_")[0].startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i not in ctrl_ids]

    # # # # # # # #
    # EXPERIMENTS #
    # # # # # # # #

    p_values = []
    for trial in TRIAL_LIST:
        this_trial = ImageTrial(DATA_ROOT, trial, "smaps")
        smap = this_trial.load_saliency_map("red")

        features_ctrl = []
        for subject in ctrl_ids:
            sub = Subject(DATA_ROOT, subject, new_res=new_res)
            out = sub.extract_fixations(trial_name=trial)
            fix_analyzer = FixationAnalyzer(DATA_ROOT, out, new_res=new_res)
            feat = fix_analyzer.average_saliency(smap)
            features_ctrl.append(feat)

        features_cvi = []
        for subject in cvi_ids:
            sub = Subject(DATA_ROOT, subject, new_res=new_res)
            out = sub.extract_fixations(trial_name=trial)
            fix_analyzer = FixationAnalyzer(DATA_ROOT, out, new_res=new_res)
            feat = fix_analyzer.average_saliency(smap)
            features_cvi.append(feat)

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
        print(trial, np.round(p_value, 4), significance)
        p_values.append(p_value)

    finalp = stats.combine_pvalues(p_values)[1]
    print("\nAfter Bonferroni correction for 55 stimuli,")
    print(f"the significant level was set to p < 0.0002,")
    print("and the combined p-value (Fisher's method)")
    print("was found at p =", finalp)
