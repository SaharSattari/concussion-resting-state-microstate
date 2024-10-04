# %% required libraries
import matplotlib.pyplot as plt
import numpy as np
import mne
import os
from mne.io import read_raw_eeglab
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind

from pycrostates.cluster import ModKMeans
from pycrostates.datasets import lemon
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks, resample

# %% Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Process accute concussed (AC_m) files
AC_m = []
files = os.listdir(data_path)
set_files = [f for f in files if f.endswith(".fif") and f.startswith("AC")]
for file in set_files:
    filepath = os.path.join(data_path, file)
    raw = mne.io.read_raw_fif(filepath, preload=True)
    AC_m.append(raw)

# Process healthy control (HC_m) files
HC_m = []
files = os.listdir(data_path)
set_files = [f for f in files if f.endswith(".fif") and f.startswith("HC")]
for file in set_files:
    filepath = os.path.join(data_path, file)
    raw = mne.io.read_raw_fif(filepath, preload=True)
    HC_m.append(raw)

# %% Generate the states

individual_gfp_peaks = list()

for group in [HC_m, AC_m]:
    for raw in group:

        gfp_peaks = extract_gfp_peaks(raw)

        # equalize peak number across subjects by resampling
        gfp_peaks = resample(gfp_peaks, n_resamples=1, n_samples=1000, random_state=42)[
            0
        ]

        individual_gfp_peaks.append(gfp_peaks.get_data())

individual_gfp_peaks = np.hstack(individual_gfp_peaks)
individual_gfp_peaks = ChData(individual_gfp_peaks, raw.info)

ModK = ModKMeans(n_clusters=7, random_state=None)
ModK.fit(individual_gfp_peaks, n_jobs=8)
ModK.plot()

# %% correct the order

ModK.reorder_clusters(order=[0,6,1,5,4,3,2])
ModK.rename_clusters(new_names=["A", "B", "C", "D", "E", "F", "G"])
ModK.plot()

# %% backfitting
ms_data = list()
subject_id = 0
for group in [HC_m, AC_m]:
    for raw in group:

        segmentation = ModK.predict(raw, factor=1, half_window_size=3)
        d = segmentation.compute_parameters()
        d["subject_id"] = subject_id
        subject_id = subject_id + 1
        ms_data.append(d)

# %% GEV plot
gev = []
total = np.zeros(len(AC_m) + len(HC_m))

for elt in ModK.cluster_names:
    for i in range(len(ms_data)):
        gev.append({"Microstate": elt, "GEV": ms_data[i][elt + "_gev"] * 100})
        total[i] = ms_data[i][elt + "_gev"] + total[i]

GEV = pd.DataFrame(gev)
# Plotting
plt.figure(figsize=(5, 4))
sns.stripplot(
    x="Microstate",
    y="GEV",
    data=GEV,
    dodge=True,
    jitter=True,
    color="black",
    size=4,
    alpha=0.7,
)
sns.boxplot(
    x="Microstate",
    y="GEV",
    data=GEV,
    width=0.4,
    linewidth=1,
    palette="Set3",
    boxprops=dict(facecolor="white", edgecolor="black"),
    showfliers=False,
)

# %% meanuduraton, occurence and coverage

data = []
for elt in ModK.cluster_names:
    for i in range(len(HC_m)):

        data.append(
            {
                "Microstate": elt,
                "Mean Duration": ms_data[i][elt + "_meandurs"] * 1000,
                "Mean Occurence": ms_data[i][elt + "_occurrences"],
                "Mean Coverage": ms_data[i][elt + "_timecov"] * 100,
                "Group_Category": "HC",
            }
        )
    for i in range(len(HC_m), len(HC_m) + len(AC_m)):

        data.append(
            {
                "Microstate": elt,
                "Mean Duration": ms_data[i][elt + "_meandurs"] * 1000,
                "Mean Occurence": ms_data[i][elt + "_occurrences"],
                "Mean Coverage": ms_data[i][elt + "_timecov"] * 100,
                "Group_Category": "AC",
            }
        )


# Convert to DataFrame
df = pd.DataFrame(data)


def plot_microstate_characteristics(df, characteristic):
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(
        x="Microstate",
        y=characteristic,
        hue="Group_Category",
        data=df,
        ci=None,
        dodge=True,
        alpha=0.8,
    )
    sns.stripplot(
        x="Microstate",
        y=characteristic,
        data=df,
        hue="Group_Category",
        ax=ax,
        dodge=True,
        jitter=True,
        marker="o",
        edgecolor="black",
        size=4,
        alpha=1,
    )
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1)
        spine.set_color("black")
    ax.legend().set_visible(False)
    ax.grid(False)
    fontdict = {"family": "Times New Roman", "size": 12, "color": "black"}
    plt.ylabel(characteristic, fontdict=fontdict)
    plt.xlabel("Microstate", fontdict=fontdict)
    plt.show()


plot_microstate_characteristics(df, "Mean Duration")
plot_microstate_characteristics(df, "Mean Occurence")
plot_microstate_characteristics(df, "Mean Coverage")

# %% statistical test


def pairwise_permutation_test(data1, data2, num_permutations=10000):
    t_observed, _ = ttest_ind(data1, data2)
    print("T-observed:", t_observed)

    combined = np.concatenate([data1, data2])
    permuted_t_stats = []

    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_group1 = combined[: len(data1)]
        perm_group2 = combined[len(data2) :]

        # Calculate t-statistic for permuted samples
        t_perm, _ = ttest_ind(perm_group1, perm_group2)
        permuted_t_stats.append(t_perm)

    p_value = (
        sum(abs(t) >= abs(t_observed) for t in permuted_t_stats) / num_permutations
    )

    # Calculate Cohen's D
    mean_diff = np.mean(data1) - np.mean(data2)
    pooled_std = np.sqrt((np.std(data1) ** 2 + np.std(data2) ** 2) / 2)
    cohen_d = mean_diff / pooled_std
    print("Cohen's D:", cohen_d)

    return p_value


for state in 'A':

    data = df[(df["Microstate"] == state) & (df["Group_Category"] == "HC")]
    array_HC = data["Mean Occurence"]
    data = df[(df["Microstate"] == state) & (df["Group_Category"] == "AC")]
    array_AC = data["Mean Occurence"]

    # Example: Comparing HC and AC
    p_value_HC_AC = pairwise_permutation_test(array_HC.values, array_AC.values)

    print(state)
    print("P-value for HC vs AC:", p_value_HC_AC)

