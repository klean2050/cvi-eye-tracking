import os, numpy as np
import sys, glob, pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from tslearn.metrics import dtw
from scipy.stats import ttest_ind

sys.path.append("./gazemae/gazemae")
from settings import *


class Subject:
    def __init__(self, root, subject_id):
        """
        Input: data path, subject id
        Function: locates and reads eye-tracking files
        """
        self.id = subject_id
        self.asc_file = root + self.id + ".asc"
        with open(self.asc_file, "r") as f:
            self.data = f.readlines()

    def get_trial_list(self, trial_name, numeric=False):
        trial_active, trial = False, []
        for line in self.data:
            if trial_active:
                if "End Trial {}".format(trial_name) in line:
                    break
                elif numeric and line[0].isdigit():
                    trial.append(line)
                elif not numeric:
                    trial.append(line)
            elif "Start Trial {}".format(trial_name) in line:
                trial_active = True
            else:
                continue
        return trial

    def get_fixations(self, left=True):
        this_eye = "L" if left else "R"
        fixation_active, fixations = False, []
        for line in self.trial:
            if fixation_active:
                if f"EFIX {this_eye}" in line:
                    fixations.append(current_fixation)
                    fixation_active = False
                else:
                    if line[0].isdigit():
                        current_fixation.append(line)
            elif f"SFIX {this_eye}" in line:
                fixation_active = True
                current_fixation = []
            else:
                continue
        return fixations

    def get_saccades(self, left=True):
        this_eye = "L" if left else "R"
        saccade_active, saccades = False, []
        for line in self.trial:
            if saccade_active:
                if f"ESACC {this_eye}" in line:
                    saccades.append(current_saccade)
                    saccade_active = False
                else:
                    if line[0].isdigit():
                        current_saccade.append(line)
            elif f"SSACC {this_eye}" in line:
                saccade_active = True
                current_saccade = []
            else:
                continue
        return saccades

    def fuse_eyes(self, array):
        # TODO: provide option to fuse eyes
        out = []
        for line in array:
            l = line.split("\t")
            row = [l[1], l[2], l[4], l[5]]
            row = [0.0 if set(r.strip()) == {"."} else float(r) for r in row]
            out.append(row)

        out = np.array(out)
        out[out < 0] = 0.0

        fused_out = []
        for (xl, yl, xr, yr) in out:
            if [xl, yl] == [0, 0]:
                fused_out.append([xr, yr])
            elif [xr, yr] == [0, 0]:
                fused_out.append([xl, yl])
            else:
                fused_out.append([(xl + xr) / 2, (yl + yr) / 2])

        fused_out = np.array(fused_out)
        fraction = 1 - np.sum((fused_out[:, 0] == 0) & (fused_out[:, 1] == 0))
        fraction /= len(fused_out)

        return fused_out, fraction

    def readTrialData(self, trial_name, vel=False):
        self.trial = self.get_trial_list(trial_name, numeric=True)
        data_fusion, fraction = self.fuse_eyes(self.trial)

        if vel:
            velocity = np.array(
                [
                    data_fusion[i + 1] - data_fusion[i]
                    for i in range(len(data_fusion) - 1)
                ]
            )
            return velocity, fraction
        else:
            return data_fusion, fraction

    def extractSaccades(self, trial_name):
        self.trial = self.get_trial_list(trial_name)
        numeric = self.get_trial_list(trial_name, numeric=True)
        if len(self.trial) <= 2:
            return []

        trial_st = int(numeric[0].split("\t")[0])
        trial_et = int(numeric[-1].split("\t")[0])

        # get both eye saccades
        saccades_l = self.get_saccades(left=True)
        saccades_r = self.get_saccades(left=False)
        self.saccades = [saccades_l, saccades_r]

        # compute the intersection between saccades
        if len(saccades_r) <= len(saccades_l):
            primary_saccades = saccades_r
            secondary_saccades = saccades_l
        else:
            primary_saccades = saccades_l
            secondary_saccades = saccades_r

        intersections = []
        for i, pri_sac in enumerate(primary_saccades):
            pri_sac_times = [d.split("\t")[0] for d in pri_sac]
            pri_sac_times = set(pri_sac_times)

            for j, sec_sac in enumerate(secondary_saccades):
                sec_sac_times = [d.split("\t")[0] for d in sec_sac]
                sec_sac_times = set(sec_sac_times)

                intersections.append(
                    [
                        i,
                        j,
                        len(list(pri_sac_times.intersection(sec_sac_times))),
                    ]
                )
        intersections.sort(key=lambda x: x[-1], reverse=True)

        saccade_idxes = intersections[: len(primary_saccades)]
        saccades_all = []
        for idx in saccade_idxes:
            a = [int(d.split("\t")[0]) for d in primary_saccades[idx[0]]]
            b = [int(d.split("\t")[0]) for d in secondary_saccades[idx[1]]]

            time_stamps = list(set(a).intersection(set(b)))
            if len(time_stamps) < 5:
                continue

            sac_st = np.min(time_stamps) - trial_st
            sac_et = np.max(time_stamps) - trial_et
            saccade_raw = [
                d
                for d in primary_saccades[idx[0]]
                if int(d.split("\t")[0]) in time_stamps
            ]

            fused_saccade, fraction = self.fuse_eyes(saccade_raw)
            saccades_all.append([fused_saccade, fraction, sac_st, sac_et])

        return saccades_all

    def extractFixations(self, trial_name):
        self.trial = self.get_trial_list(trial_name)
        numeric = self.get_trial_list(trial_name, numeric=True)
        if len(self.trial) <= 2:
            return []

        trial_st = int(numeric[0].split("\t")[0])
        trial_et = int(numeric[-1].split("\t")[0])

        # get both eye fixations
        fixations_l = self.get_fixations(left=True)
        fixations_r = self.get_fixations(left=False)
        self.fixations = [fixations_l, fixations_r]

        # compute the intersection between fixations
        if len(fixations_r) <= len(fixations_l):
            primary_fixations = fixations_r
            secondary_fixations = fixations_l
        else:
            primary_fixations = fixations_l
            secondary_fixations = fixations_r

        intersections = []
        for i, pri_fix in enumerate(primary_fixations):
            pri_fix_times = [d.split("\t")[0] for d in pri_fix]
            pri_fix_times = set(pri_fix_times)

            for j, sec_fix in enumerate(secondary_fixations):
                sec_fix_times = [d.split("\t")[0] for d in sec_fix]
                sec_fix_times = set(sec_fix_times)

                intersections.append(
                    [
                        i,
                        j,
                        len(list(pri_fix_times.intersection(sec_fix_times))),
                    ]
                )
        intersections.sort(key=lambda x: x[-1], reverse=True)

        fixation_idxes = intersections[: len(primary_fixations)]
        fixations_all = []
        for idx in fixation_idxes:
            a = [int(d.split("\t")[0]) for d in primary_fixations[idx[0]]]
            b = [int(d.split("\t")[0]) for d in secondary_fixations[idx[1]]]
            time_stamps = list(set(a).intersection(set(b)))
            if len(time_stamps) < 10:
                break

            fixation_st = np.min(time_stamps) - trial_st
            fixation_et = np.max(time_stamps) - trial_et
            fixation_raw = [
                d
                for d in primary_fixations[idx[0]]
                if int(d.split("\t")[0]) in time_stamps
            ]

            fused_fixation, fraction = self.fuse_eyes(fixation_raw)
            fixations_all.append([fused_fixation, fraction, fixation_st, fixation_et])

        return fixations_all


class DTWAnalysis:
    def readAllData(self, trial_name):
        data_dir = "../asc_data_v1"
        subject_ids = glob.glob(os.path.join(data_dir, "*.asc"))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        self.timeseries = {}
        self.data_frac = {}
        for subject in subject_ids:
            sub = Subject(subject)
            trial_data, frac = sub.readTrialData(trial_name, vel=self.vel)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - frac

    def plotDistanceMatrixTrialWise(self, trial_name, vel=False):
        print(f"distance matrix for  {trial_name}")
        self.vel = vel
        self.readAllData(trial_name)
        subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
        subject_ids.sort()
        distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
        for i, keyi in enumerate(subject_ids):
            for j, keyj in enumerate(subject_ids):
                distance_matrix[i, j] = dtw(
                    self.timeseries[keyi][:1000], self.timeseries[keyj][:1000]
                )
        plt.clf()
        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(distance_matrix)
        ax.set_xticks(list(range(len(subject_ids))))
        ax.set_xticklabels(subject_ids, rotation=90)
        fig.colorbar(img)
        plt.tight_layout()
        if self.vel:
            plt.savefig(f"distance_matrix_vel/{trial_name}.png")
        else:
            plt.savefig(f"distance_matrix/{trial_name}.png")


class SaliencyTrace:
    def __init__(self, root, smaps=["all"]):
        self.root = root
        self.smap_dir = "cvi-extra/saliency_maps/"
        if "all" in smaps:
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
            self.smaps = smaps

    def fix_bounds(self, data):
        data = [d for d in data if (d[0] != 0) or (d[1] != 0)]
        for i, (x, y) in enumerate(data):
            x = 1279 if x >= 1280 else 0 if x < 0 else x
            y = 719 if y >= 720 else 0 if y < 0 else y
            data[i] = [int(x), int(y)]
        return data

    def readAllData(self, trial_name):
        subject_ids = glob.glob(os.path.join(self.root, "*.asc"))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]

        self.timeseries, self.data_frac = {}, {}
        for subject in subject_ids:
            sub = Subject(self.root, subject)
            trial_data, fraction = sub.readTrialData(trial_name)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - fraction

    def readAllFixations(self, trial_name):
        subject_ids = glob.glob(os.path.join(self.root, "*.asc"))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]

        self.fixation_timeseries = {}
        for subject in subject_ids:
            sub = Subject(self.root, subject)
            fixations = sub.extractFixations(trial_name)
            self.fixation_timeseries[subject] = fixations

    def computeTraces(self, trial_name, data):
        name = trial_name[:-4]
        traces = {}
        for smap in self.smaps:
            if smap == "grad":
                smap_path = os.path.join(self.smap_dir, "gen", f"{trial_name}.npy")
            else:
                smap_path = os.path.join(self.smap_dir, f"{name}/{name}_{smap}.npy")
            try:
                saliency_map = np.load(smap_path).squeeze()
            except:
                print(f"SKIPPING {name} {smap}")
                traces[smap] = -1
                continue

            traces[smap] = [saliency_map[d[1], d[0]] for d in self.fix_bounds(data)]
        return traces

    def computeFixationTraces(self, trial_name, fixations):
        self.readAllFixations(trial_name)
        smap_path = os.path.join(self.smap_dir, "gen", f"{trial_name}.npy")
        saliency_map = np.load(smap_path).squeeze()
        saliency_map -= np.min(saliency_map[:])
        saliency_map /= np.max(saliency_map) - np.min(saliency_map)

        avg_fixations = []
        for (fixation, frac, st, et) in fixations:
            fixation = [d for d in fixation if (d[0] != 0) or (d[1] != 0)]
            fixation = self.fix_bounds(fixation)
            trace_ = [saliency_map[d[1], d[0]] for d in fixation]
            avg_fixations.append([np.mean(trace_), frac, st, et])

        return avg_fixations

    def computeFixationTraceForAll(self, trial_name):
        self.readAllFixations(trial_name)
        self.avg_fixations = {}
        for sub, data in self.fixation_timeseries.items():
            self.avg_fixations[sub] = self.computeFixationTraces(trial_name, data)

    def computeTraceForALL(self, trial_name):
        self.readAllData(trial_name)
        subject_ids = [k for k in self.data_frac.keys() if self.data_frac[k] > 0.5]
        subject_ids.sort()

        self.traces = {}
        for subject in subject_ids:
            self.traces[subject] = self.computeTraces(
                trial_name, self.timeseries[subject]
            )

    def plotTrace(self, trial_name):
        self.computeTraceForALL(trial_name)
        plt.clf()
        for subject, trace in self.trace.items():
            x = [i for i in range(1, len(trace) + 1)]
            plt.plot(x, trace, label=subject)
        plt.grid()
        plt.legend()
        plt.savefig("trace_all.png", dpi=300)

    def computeDistance(self, trial_name):
        self.computeTraceForALL(trial_name)
        subject_ids = list(self.traces.keys())
        subject_ids.sort()

        distance_matrix = np.empty((len(subject_ids), len(subject_ids)))
        for smap in tqdm(self.smaps):
            for i, keyi in enumerate(subject_ids):
                for j, keyj in enumerate(subject_ids):
                    distance_matrix[i, j] = dtw(
                        self.traces[keyi][smap][:1000], self.traces[keyj][smap][:1000]
                    )

            plt.clf()
            fig, ax = plt.subplots(1, 1)
            img = ax.imshow(distance_matrix)
            ax.set_xticks(list(range(len(subject_ids))))
            ax.set_xticklabels(subject_ids, rotation=90)
            fig.colorbar(img)
            plt.tight_layout()
            save_dir = f"output/distance_dtw_trace_smaps/{trial_name}"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                f"{save_dir}/{trial_name}_{smap}.png",
                dpi=300,
            )

    def compareTrials(self):
        def computeTrace(self, trial_name, data):
            smap_path = os.path.join(self.smap_dir, "gen", f"{trial_name}.npy")
            saliency_map = np.load(smap_path).squeeze()
            saliency_map -= np.min(saliency_map[:])
            saliency_map /= np.max(saliency_map) - np.min(saliency_map)
            return [saliency_map[d[1], d[0]] for d in self.fix_bounds(data)]

        subject_ids = glob.glob(os.path.join(self.root, "*.asc"))
        subject_ids = [os.path.basename(d)[:-4] for d in subject_ids]
        compare_trials = [
            ["Freeviewingstillimage_36.jpg", "Freeviewingstillimage_36_cutout.tif"],
            ["Freeviewingstillimage_28.jpg", "Freeviewingstillimage_28_cutout.tif"],
            ["Freeviewingstillimage_93.jpg", "Freeviewingstillimage_93_cutout.tif"],
            ["Freeviewingstillimage_36.jpg", "Freeviewingstillimage_36_cutout.tif"],
            ["Freeviewingstillimage_10.jpg", "Freeviewingstillimage_10_cutout.tif"],
        ]
        for pair in compare_trials:
            comparison = {}
            for subject in subject_ids:
                sub = Subject(subject)
                data0, frac0 = sub.readTrialData(pair[0])
                data1, frac1 = sub.readTrialData(pair[1])
                if (frac0 < 0.5) and (frac1 < 0.5):
                    trace0 = computeTrace(pair[0], data0)
                    trace1 = computeTrace(pair[1], data1)
                    comparison[subject] = dtw(trace0, trace1)

            x_y = [[k, v] for k, v in comparison.items()]
            x_y.sort(key=lambda x: x[0])

            plt.clf()
            plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
            plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
            plt.grid()
            plt.tight_layout()
            name = pair[0][:-4].split("_")[1]
            plt.savefig(f"comparison_{name}.png")


class TraceAnalyzer:
    def __init__(self, trial_name, smaps=["all"]) -> None:
        self.trial_name = trial_name
        self.st = SaliencyTrace(smaps=smaps)

    def representation(self, type="avg"):
        self.st.computeTraceForALL(self.trial_name)
        self.traces = self.st.traces
        # dict = {}
        # for sub in self.traces.keys():
        #     dict[sub] = np.mean(self.traces[sub])
        # return dict
        # return {sub: np.mean(self.traces[sub] for sub in self.traces.keys())}

    def metrics_all_saliency(self):
        self.representation()
        done = [d.rstrip().split(" ")[0] for d in open("ttest_trace_avg_smaps.txt")]
        for smap in self.st.smaps:
            if f"{self.trial_name[:-4]}_{smap}" in done:
                continue
            data = [
                [sub, np.mean(traces[smap])]
                for sub, traces in self.traces.items()
                if traces[smap] != -1
            ]
            data.sort(key=lambda x: x[0])
            # print(data)
            plt.clf()
            plt.bar(list(range(len(data))), [d[1] for d in data])
            plt.xticks(list(range(len(data))), [d[0] for d in data], rotation=90)
            plt.grid()
            plt.tight_layout()
            dir = os.path.join("trace_rep_smaps", self.trial_name[:-4])
            os.makedirs(dir, exist_ok=True)
            save_name = os.path.join(dir, f"{self.trial_name[:-4]}_{smap}.png")
            plt.savefig(save_name, dpi=300)
            a = [d[1] for d in data if d[0].startswith("1")]
            b = [d[1] for d in data if d[0].startswith("2")]
            with open("ttest_trace_avg_smaps.txt", "a") as f:
                print(f"{self.trial_name[:-4]}_{smap}", ttest_ind(a, b), file=f)

    def metrics(self):
        rep = self.representation()
        x_y = [[k, v] for k, v in rep.items()]
        x_y.sort(key=lambda x: x[0])
        plt.clf()
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        name = os.path.join("trace_rep", self.trial_name[:-4] + ".png")
        plt.savefig(name, dpi=300)
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        with open("ttest_trace_avg.txt", "a") as f:
            print(self.trial_name[:-4], ttest_ind(a, b), file=f)

    def plotTraces(self):
        subjects = list(self.traces.keys())
        cvi = [sub for sub in subjects if sub.startswith("1")]
        ctrl = [sub for sub in subjects if sub.startswith("2")]
        plt.clf()
        for sub in ctrl:
            x = [i for i in range(1, len(self.traces[sub]) + 1)]
            plt.plot(x, self.traces[sub], label=sub)
        plt.savefig(f"trace_group_wise/{self.trial_name[:-4]}_ctrl.png", dpi=300)
        plt.clf()
        for sub in cvi:
            x = [i for i in range(1, len(self.traces[sub]) + 1)]
            plt.plot(x, self.traces[sub], label=sub)
        plt.savefig(f"trace_group_wise/{self.trial_name[:-4]}_cvi.png", dpi=300)


class FixationAnalyzer:
    def __init__(self, root, trial_name):
        self.root = root
        self.trial_name = trial_name
        self.avg_fixation = pickle.load(
            open(
                os.path.join(self.root, "avg_fixation_trial_wise", trial_name + ".pkl"),
                "rb",
            )
        )

    def latency_first_fixation(self):
        x_y = [[sub, data[0][2]] for sub, data in self.avg_fixation.items()]
        x_y.sort(key=lambda x: x[0])
        return x_y

    def saliency_first_fixation(self):
        x_y = [[sub, data[0][0]] for sub, data in self.avg_fixation.items()]
        x_y.sort(key=lambda x: x[0])
        return x_y

    def saliency_longest_fixation(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key=lambda x: x[-1] - x[-2])
            x_y.append([sub, data[-1][0]])
        x_y.sort(key=lambda x: x[0])
        return x_y

    def latency_longest_fixation(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key=lambda x: x[-1] - x[-2])
            x_y.append([sub, data[-1][2]])
        x_y.sort(key=lambda x: x[0])
        return x_y

    def latency_maximum_saliency(self):
        x_y = []
        for sub, data in self.avg_fixation.items():
            data.sort(key=lambda x: x[0])
            x_y.append([sub, data[-1][2]])
        x_y.sort(key=lambda x: x[0])
        return x_y

    def compute_metrics(self):
        stats = {}
        plt.clf()
        x_y = self.latency_first_fixation()
        plt.subplot(2, 3, 1)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title("latency_first_fixation")
        plt.ylabel("time")
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        stat, p_value = ttest_ind(a, b)
        stats["latency_first_fixation"] = [stat, p_value]

        x_y = self.saliency_first_fixation()
        plt.subplot(2, 3, 2)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title("saliency_first_fixation")
        plt.ylabel("time")
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        stat, p_value = ttest_ind(a, b)
        stats["saliency_first_fixation"] = [stat, p_value]

        x_y = self.saliency_longest_fixation()
        plt.subplot(2, 3, 3)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title("saliency_longest_fixation")
        plt.ylabel("time")
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        stat, p_value = ttest_ind(a, b)
        stats["saliency_longest_fixation"] = [stat, p_value]

        x_y = self.latency_longest_fixation()
        plt.subplot(2, 3, 4)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title("latency_longest_fixation")
        plt.ylabel("time")
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        stat, p_value = ttest_ind(a, b)
        stats["latency_longest_fixation"] = [stat, p_value]

        x_y = self.latency_maximum_saliency()
        plt.subplot(2, 3, 5)
        plt.bar(list(range(len(x_y))), [d[1] for d in x_y])
        plt.xticks(list(range(len(x_y))), [d[0] for d in x_y], rotation=90)
        plt.grid()
        plt.tight_layout()
        plt.title("latency_maximum_saliency")
        plt.ylabel("time")
        a = [d[1] for d in x_y if d[0].startswith("1")]
        b = [d[1] for d in x_y if d[0].startswith("2")]
        stat, p_value = ttest_ind(a, b)
        stats["latency_maximum_saliency"] = [stat, p_value]
        plt.savefig("output/fixation_stats/" + self.trial_name[:-4] + ".png", dpi=300)
        return stats


if __name__ == "__main__":
    root = "/home/kavra/Datasets/medical/cvi_eyetracking/asc_data_v1/"
    trial, vel = "Freeviewingstillimage_1.jpg", False

    cvi_keys = [
        "1007_4",
        "1007_1",
        "1008_1",
        "1003_3",
        "1018_2",
        "1003_2",
        "1005_1",
        "1017_2",
        "1007_3",
    ]
    ctrl_keys = ["2003_1", "2002_2", "2002_1", "2004_2", "2004_1", "2006_1"]

    sub = Subject(root, "1003_3")
    data, fr = sub.readTrialData(trial, vel)
    allfix = sub.extractFixations(trial)
    allsac = sub.extractSaccades(trial)

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

    # for trial_name in tqdm(trials_images_subset):
    # print(trial_name)
    # fr.plotDistanceMatrix(trial_name, vel=True)
    #     # da.plotDistanceMatrixTrialWise(trial_name)
    # st.computeDistance(trial_name)
    # st.plotTrace(trial_name)
    # ta = TraceAnalyzer(trial_name)
    # ta.metrics_all_saliency()
    # break
    # ta.plotTraces()'

    # save the avg fixation
    # print(trial_name)
    # file_name =os.path.join('avg_fixation_trial_wise', trial_name + '.pkl')
    # if os.path.isfile(file_name):
    #     continue
    # st = SaliencyTrace()
    # st.computeFixationTraceForAll(trial_name)
    # f = open(os.path.join('avg_fixation_trial_wise', trial_name + '.pkl'), 'wb')
    # pkl.dump(st.avg_fixations, f)
    # f.close()

    # analyze the saved fixation
    # stats = {}
    # print(trial_name)
    # fa =FixationAnalyzer(trial_name)
    # stats[trial_name[:-4]] = fa.compute_metrics()

    # f =open('avg_fixation_stats_all_trials.pkl', 'wb')
    # pkl.dump(stats, f)
    # f.close()
