---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
orphan: true
---

# Question 2: Pairwise Spike Train Interactions

**In which brain area are pairwise spike train interactions strongest at the 100 ms timescale?**

## Approach

We compute pairwise cross-correlograms between spike trains within each brain area and use the mean cross-correlogram value as a proxy for interaction strength. Statistical comparisons across areas use the Kruskal-Wallis test, with post-hoc Mann-Whitney U tests and Bonferroni correction.

## Setup

```{code-cell} ipython3
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import lindi
import pynapple as nap
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

dandiset_id = "218201"
```

## Cross-correlograms per brain area

For each dataset, we load spike trains and compute normalized pairwise cross-correlograms within each brain area using a 10 ms bin and a ±100 ms window. The mean value across all pairs serves as the area-level interaction score (AUC).

```{code-cell} ipython3
auc = {}

for dataset_num in range(1, 5):

    filepath = f"sub-mouse-{dataset_num}/sub-mouse-{dataset_num}_ses-None_ecephys.nwb"
    with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = lindi.LindiH5pyFile.from_hdf5_file(s3_url, local_cache=lindi.LocalCache())
    io = NWBHDF5IO(file=f)
    spikes = nap.NWBFile(io.read())["units"]

    auc[dataset_num] = []
    for area in [1, 2, 3]:
        cc = nap.compute_crosscorrelogram(
            spikes[spikes.brain_area == area],
            binsize=0.01,
            windowsize=0.1,
            norm=True
        )
        auc[dataset_num].append(cc.loc[-0.1:0.1].mean(0).values)
```

## Kruskal-Wallis test

We test whether interaction strength differs significantly across the three brain areas using the non-parametric Kruskal-Wallis test.

```{code-cell} ipython3
    kruskal_result = kruskal(*auc[dataset_num])
    print(f"Dataset {dataset_num}: Kruskal-Wallis p-value = {kruskal_result.pvalue:.4f}")
```

## Post-hoc pairwise comparisons

When the Kruskal-Wallis test is significant, we run Mann-Whitney U tests for each pair of areas and apply Bonferroni correction for multiple comparisons.

```{code-cell} ipython3
    area_pairs = list(combinations([1, 2, 3], 2))
    n_comparisons = len(area_pairs)

    for area1, area2 in area_pairs:
        u_stat, p_val = mannwhitneyu(auc[dataset_num][area1 - 1], auc[dataset_num][area2 - 1])
        p_corrected = min(p_val * n_comparisons, 1.0)
        if kruskal_result.pvalue < 0.05:
            print(
                f"  Post-hoc area {area1} vs area {area2}: "
                f"U={u_stat:.1f}, p={p_val:.4f}, p_corrected={p_corrected:.4f}"
            )
```

## Collecting results

We store per-dataset statistics — Kruskal-Wallis results, mean AUC per area, and all pairwise test outcomes — into a DataFrame.

```{code-cell} ipython3
    rows = []

    row = {
        "dataset": dataset_num,
        "kruskal_statistic": kruskal_result.statistic,
        "kruskal_pvalue": kruskal_result.pvalue,
        "kruskal_significant": kruskal_result.pvalue < 0.05,
        "mean_auc_area1": np.mean(auc[dataset_num][0]),
        "mean_auc_area2": np.mean(auc[dataset_num][1]),
        "mean_auc_area3": np.mean(auc[dataset_num][2]),
        "strongest_area": int(np.argmax([np.mean(a) for a in auc[dataset_num]]) + 1),
    }

    for area1, area2 in area_pairs:
        u_stat, p_val = mannwhitneyu(auc[dataset_num][area1 - 1], auc[dataset_num][area2 - 1])
        p_corrected = min(p_val * n_comparisons, 1.0)
        col_prefix = f"area{area1}_vs_area{area2}"
        row[f"{col_prefix}_u"] = u_stat
        row[f"{col_prefix}_p"] = p_val
        row[f"{col_prefix}_p_corrected"] = p_corrected
        row[f"{col_prefix}_significant"] = p_corrected < 0.05

    rows.append(row)

results_df = pd.DataFrame(rows).set_index("dataset")
print(results_df.to_string())
```

## Summary

We tally which area is the strongest most frequently across datasets, and compute the overall mean AUC per area.

```{code-cell} ipython3
strongest_counts = results_df["strongest_area"].value_counts()
print(strongest_counts.to_string())

overall_mean = {
    1: results_df["mean_auc_area1"].mean(),
    2: results_df["mean_auc_area2"].mean(),
    3: results_df["mean_auc_area3"].mean(),
}
best_area = max(overall_mean, key=overall_mean.get)
print(f"Overall mean AUC — Area 1: {overall_mean[1]:.4f}, Area 2: {overall_mean[2]:.4f}, Area 3: {overall_mean[3]:.4f}")
```

## Answer

```{code-cell} ipython3
print(f"Brain area {best_area} has the strongest pairwise spike train interactions at the 100 ms timescale.")
```