import os
from scipy import stats
from process import FixationAnalyzer, Subject
from utils import DATA_ROOT, TRIAL_LIST

if __name__ == "__main__":

    ids = [i for i in os.listdir(DATA_ROOT) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.split("_")[0].startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i not in ctrl_ids]
    names = ctrl_ids + cvi_ids

    # # # # # # # # # # #
    # SAMPLE EXPERIMENT #
    # # # # # # # # # # #

    for trial in TRIAL_LIST:
        features_ctrl = []
        for subject in ctrl_ids:
            sub = Subject(DATA_ROOT, subject)
            out = sub.extract_fixations(trial_name=trial)
            fix_analyzer = FixationAnalyzer(DATA_ROOT, out)
            dur, _ = fix_analyzer.duration_of_fixations()
            features_ctrl.append(dur)

        features_cvi = []
        for subject in cvi_ids:
            sub = Subject(DATA_ROOT, subject)
            out = sub.extract_fixations(trial_name=trial)
            fix_analyzer = FixationAnalyzer(DATA_ROOT, out)
            dur, _ = fix_analyzer.duration_of_fixations()
            features_cvi.append(dur)

        stat, p_value = stats.ttest_ind(features_ctrl, features_cvi)
        significance = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else ""
        )
        print(trial, p_value, significance)
