import numpy as np
import matplotlib.pyplot as plt
from process import Subject
from tslearn.metrics import dtw

DATA_ROOT1 = "/PLACEHOLDER/"
DATA_ROOT2 = "/PLACEHOLDER/"
IMGS_ROOT = "trials/"

TRIAL_LIST = [
    "Freeviewingstillimage_1",
    "Freeviewingstillimage_2",
    "Freeviewingstillimage_4",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_7",
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_10_cutout",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_12",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_17",
    "Freeviewingstillimage_18",
    "Freeviewingstillimage_19",
    "Freeviewingstillimage_20",
    "Freeviewingstillimage_21",
    "Freeviewingstillimage_22",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_24",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_27",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_28_cutout",
    "Freeviewingstillimage_29",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_35",
    "Freeviewingstillimage_36",
    "Freeviewingstillimage_36_cutout",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_40",
    "Freeviewingstillimage_41",
    "Freeviewingstillimage_45",
    "Freeviewingstillimage_46",
    "Freeviewingstillimage_47",
    "Freeviewingstillimage_88",
    "Freeviewingstillimage_92",
    "Freeviewingstillimage_93",
    "Freeviewingstillimage_93_cutout",
    "Moviestillimage_6",
    "Moviestillimage_8",
]

VISUAL_SEARCH = [
    "visual search form 4_1",
    "visual search form 8_1",
    "visual search form 16_1",
    "visual search form 24_1",
    "visual search form 32_1",
    "visual search orientation 4_1",
    "visual search orientation 8_1",
    "visual search orientation 16_1",
    "visual search orientation 24_1",
    "visual search orientation 32_1",
]

CUTOUTS = [
    ["Freeviewingstillimage_10", "Freeviewingstillimage_10_cutout"],
    ["Freeviewingstillimage_28", "Freeviewingstillimage_28_cutout"],
    ["Freeviewingstillimage_36", "Freeviewingstillimage_36_cutout"],
    ["Freeviewingstillimage_93", "Freeviewingstillimage_93_cutout"],
]

TRIAL_TEXTURE = [
    "Freeviewingstillimage_1",
    "Freeviewingstillimage_4",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_22",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_40",
]

TRIAL_COMPLEXITY = [
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_24",
    "Freeviewingstillimage_27",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_46",
]

TRIAL_ORIENTATION = [
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_24",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_40",
    "Freeviewingstillimage_41",
]

TRIAL_BRIGHTNESS = [
    "Freeviewingstillimage_1",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_19",
    "Freeviewingstillimage_21",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_27",
    "Freeviewingstillimage_29",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_36",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_40",
    "Freeviewingstillimage_45",
    "Freeviewingstillimage_46",
    "Freeviewingstillimage_47",
]

TRIAL_BRIGHT_COLORS = [
    "Freeviewingstillimage_2",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_21",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_24",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_27",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_47",
]

TRIAL_COLOR = [
    "Freeviewingstillimage_2",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_17",
    "Freeviewingstillimage_22",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_24",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_27",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_41",
    "Freeviewingstillimage_47",
    "Freeviewingstillimage_92",
]

TRIAL_DEPTH = [
    "Freeviewingstillimage_1",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_18",
    "Freeviewingstillimage_20",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_29",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_36",
    "Freeviewingstillimage_39",
]

TRIAL_FACE = [
    "Freeviewingstillimage_2",
    "Freeviewingstillimage_4",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_7",
    "Freeviewingstillimage_8",
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_17",
    "Freeviewingstillimage_19",
    "Freeviewingstillimage_20",
    "Freeviewingstillimage_23",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_45",
]

TRIAL_FACE_INTERACTION = [
    "Freeviewingstillimage_7",
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_11",
    "Freeviewingstillimage_12",
    "Freeviewingstillimage_13",
    "Freeviewingstillimage_15",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_21",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_28",
    "Freeviewingstillimage_29",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_45",
    "Freeviewingstillimage_47",
    "Freeviewingstillimage_93",
]

TRIAL_FACE_HANDS = [
    "Freeviewingstillimage_2",
    "Freeviewingstillimage_5",
    "Freeviewingstillimage_7",
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_20",
    "Freeviewingstillimage_21",
    "Freeviewingstillimage_25",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_39",
    "Freeviewingstillimage_45",
]

TRIAL_EYES_MOUTH = [
    "Freeviewingstillimage_9",
    "Freeviewingstillimage_17",
    "Freeviewingstillimage_20",
    "Freeviewingstillimage_26",
    "Freeviewingstillimage_33",
    "Freeviewingstillimage_39",
]

TRIAL_MOVEMENT = [
    "Freeviewingstillimage_7",
    "Freeviewingstillimage_10",
    "Freeviewingstillimage_16",
    "Freeviewingstillimage_31",
    "Freeviewingstillimage_35",
    "Freeviewingstillimage_36",
    "Freeviewingstillimage_45",
    "Freeviewingstillimage_47",
    "Freeviewingstillimage_88",
    "Moviestillimage_6",
    "Moviestillimage_8",
]


def plot_traces(names, traces, outpath):
    t = list(range(len(names)))
    plt.figure(figsize=(10, 5))
    for subject, trace in zip(names, traces):
        plt.plot(t, trace, label=subject)

    plt.xticks(t, names, rotation=90)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.legend()
    plt.title("Fixation Traces")
    plt.savefig(outpath, dpi=300)


def plot_stats(names, stats, title, outpath):
    t = list(range(len(names)))
    plt.figure(figsize=(10, 5))
    plt.bar(names, stats)
    plt.xticks(t, names, rotation=90)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.title(title)
    plt.savefig(outpath)


def plot_distance_matrix(names, dmatrix, outpath):
    t = list(range(len(names)))
    fig, ax = plt.figure(figsize=(10, 5))
    img = ax.imshow(dmatrix)
    ax.set_xticks(t)
    ax.set_xticklabels(names, rotation=90)
    fig.colorbar(img)
    plt.tight_layout()
    plt.title("Eyetrack Distance Matrix")
    plt.savefig(outpath)


def compute_distances(names, lists, trial_name):
    distance_matrix = np.empty((len(names), len(names)))
    for i, ki in enumerate(names):
        for j, kj in enumerate(names):
            ln = min(len(lists[ki]), len(lists[kj]))
            distance_matrix[i, j] = dtw(lists[ki][:ln], lists[kj][:ln])

    outpath = f"output/dist_trace_{trial_name}.png"
    plot_distance_matrix(names, distance_matrix, outpath)


def compare_trial_traces(names, trials, outpath):
    comparison = {}
    for subject in names:
        sub = Subject(subject)
        data0, frac0 = sub.extract_data(trials[0])
        data1, frac1 = sub.extract_data(trials[1])
        if (frac0 < 0.5) and (frac1 < 0.5):
            trace0 = sub.compute_trace(trials[0], data0)
            trace1 = sub.compute_trace(trials[1], data1)
            comparison[subject] = dtw(trace0, trace1)

    compared = [comparison[k] for k in comparison.keys()]
    title = f"DTW Between {trials[0]} and {trials[1]}"
    plot_stats(names, compared, title, outpath)

    x_y = [[k, v] for k, v in comparison.items()]
    x_y.sort(key=lambda x: x[0])


def fix_bounds(data, new_res=False):
    new_data = []
    for x, y in data:
        if new_res:
            x = np.nan if x >= 1279 + 320 else np.nan if x <= 319 else x
            y = np.nan if y >= 719 + 240 else np.nan if y <= 239 else y
        else:
            x = np.nan if x >= 1279 else np.nan if x <= 0 else x
            y = np.nan if y >= 719 else np.nan if y <= 0 else y
        try:
            new_data.append([int(x), int(y)])
        except:
            new_data.append([np.nan, np.nan])
    return np.array(new_data)
