import matplotlib.pyplot as plt
import numpy as np, seaborn as sns
from scipy import stats


def statistic_mw(x, y):
    return stats.mannwhitneyu(x, y)[0]


def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    return q3 - q1


def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / (
        np.sqrt(
            ((len(x) - 1) * np.std(x) ** 2 + (len(y) - 1) * np.std(y) ** 2)
            / (len(x) + len(y) - 2)
        )
    )


def plot_distribution(ctrl_features, cvi_features, name):
    fig, ax = plt.subplots(figsize=(3.5, 4))
    sns.swarmplot(
        data=[ctrl_features, cvi_features],
        palette=["#A1C9F4", "#5fc972"],
        ax=ax,
        size=5,
        alpha=0.5,
    )
    sns.boxplot(
        data=[ctrl_features, cvi_features],
        palette=["#73afef", "#4dbf61"],
        ax=ax,
        width=0.3,
        # shorten distance between the two boxplots
        showfliers=False,
    )
    ax.set_xticklabels(["CTRL", "CVI"], fontsize=14)
    ax.set_ylabel(name, fontsize=12)
    sns.despine(bottom=True)
    plt.title("", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
