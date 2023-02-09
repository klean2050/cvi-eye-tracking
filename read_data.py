import os, numpy as np
import sys, glob, pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from tslearn.metrics import dtw
from scipy.stats import ttest_ind

from subject import Subject


class SaliencyTrace:
    def __init__(self, root, smap_dir, smaps=["all"]):
        self.root = root
        self.ids = glob.glob(os.path.join(self.root, "*.asc"))
        self.ids = [os.path.basename(d)[:-4] for d in self.ids]
        self.smap_dir = smap_dir  # "cvi-extra/saliency_maps/"
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

    def read_data_all(self, trial_name, vel=False):
        self.timeseries, self.data_frac = {}, {}
        for subject in self.ids:
            sub = Subject(self.root, subject)
            trial_data, fraction = sub.extract_data(trial_name, vel)
            self.timeseries[subject] = trial_data
            self.data_frac[subject] = 1 - fraction

    def read_fixations_all(self, trial_name):
        self.fixation_timeseries = {}
        for subject in self.ids:
            sub = Subject(self.root, subject)
            fixations = sub.extract_fixations(trial_name)
            self.fixation_timeseries[subject] = fixations

    def compute_data_traces(self, trial_name, data):
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

    def compute_fixation_traces(self, trial_name, fixations):
        self.read_fixations_all(trial_name)
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

    def compute_avg_fixations_all(self, trial_name):
        self.read_fixations_all(trial_name)
        self.avg_fixations = {}
        for sub, data in self.fixation_timeseries.items():
            self.avg_fixations[sub] = self.compute_fixation_traces(trial_name, data)

    def compute_trace_all(self, trial_name):
        self.read_data_all(trial_name)
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
                data0, frac0 = sub.extract_data(pair[0])
                data1, frac1 = sub.extract_data(pair[1])
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


if __name__ == "__main__":
    root = "/home/kavra/Datasets/medical/cvi_eyetracking/asc_data_v1/"
    trial, vel = "Freeviewingstillimage_1.jpg", False

    sub = Subject(root, "1003_3")
    data, fr = sub.extract_data(trial, vel)
