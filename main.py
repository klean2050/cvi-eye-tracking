import os
from scipy import stats
from process import FixationAnalyzer, Subject
from utils import DATA_ROOT

if __name__ == "__main__":
    root = DATA_ROOT
    trial = "Freeviewingstillimage_1.jpg"

    ids = [i for i in os.listdir(DATA_ROOT) if i.endswith(".asc")]
    ctrl_ids = [i.split(".")[0] for i in ids if i.split("_")[0].startswith("2")]
    cvi_ids = [i.split(".")[0] for i in ids if i not in ctrl_ids]
    names = ctrl_ids + cvi_ids

    # # # # # # # # # # #
    # SAMPLE EXPERIMENT #
    # # # # # # # # # # #

    durations_ctrl = []
    for subject in ctrl_ids:
        sub = Subject(DATA_ROOT, subject)
        out = sub.extract_fixations(trial_name=trial)
        fix_analyzer = FixationAnalyzer(DATA_ROOT, out)
        dur = fix_analyzer.number_of_fixations()
        durations_ctrl.append(dur)

    durations_cvi = []
    for subject in cvi_ids:
        sub = Subject(DATA_ROOT, subject)
        out = sub.extract_fixations(trial_name=trial)
        fix_analyzer = FixationAnalyzer(DATA_ROOT, out)
        dur = fix_analyzer.number_of_fixations()
        durations_cvi.append(dur)

    stat, p_value = stats.ttest_ind(durations_ctrl, durations_cvi)
    print(p_value)
