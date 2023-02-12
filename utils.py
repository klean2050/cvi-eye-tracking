import matplotlib.pyplot as plt
import numpy as np
from tslearn.metrics import dtw

from process.subject import Subject


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
