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

# Question 3: Functional Connectivity Between Brain Areas

**Which brain area pair has the strongest directed functional connectivity?**

## Approach

We fit Poisson GLMs to predict the activity of one brain area from another and compare predictive performance as
a proxy for functional connectivity.

## Setup

```{code-cell} ipython3
import warnings
from pathlib import Path

import jax
import lindi
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pandas as pd
import pynapple as nap
import seaborn as sns
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Converting 'd' to numpy.array.*")
jax.config.update("jax_enable_x64", True)

dandiset_id = "218201"
np.random.seed(42)

window_size_sec = 0.8
bin_size_sec = 0.02
n_basis_funcs = 4
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=n_basis_funcs,
    window_size=int(window_size_sec // bin_size_sec),
)

train_set = nap.IntervalSet([0, 60])
test_set = nap.IntervalSet([60, 120])
```

## Population-level Poisson GLMs

For each dataset, we load spike trains and fit a population-level GLM for every possible pair of brain areas, including self-prediction.

### Notes

- We subsample the units to the lowest number per brain area, otherwise we might interpret performance differences caused by different numbers of units as functional connectivity.
- We use NeMoS' `GroupLasso` to regularize each unit's features together.
- We filter out units with a firing rate below 1 Hz to avoid silent neuron issues during fitting.

```{code-cell} ipython3
results = []
best_pairs = []

for dataset_num in [9]:

    # Stream data from DANDI
    filepath = f"sub-mouse-{dataset_num}/sub-mouse-{dataset_num}_ses-None_ecephys.nwb"
    with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = lindi.LindiH5pyFile.from_hdf5_file(s3_url, local_cache=lindi.LocalCache())
    io = NWBHDF5IO(file=f)
    units = nap.NWBFile(io.read())["units"]
    units = units[units.rate > 1.0]

    counts = units.count(bin_size_sec)
    subsample_n = units["brain_area"].value_counts().min()

    session_results = []

    for predicted_area in tqdm(
        units["brain_area"].unique(), desc="predicted area", leave=False
    ):
        predicted_area_counts = counts[:, units["brain_area"] == predicted_area]
        predicted_area_counts = predicted_area_counts[
            :,
            np.random.choice(
                predicted_area_counts.shape[1], subsample_n, replace=False
            ),
        ]

        for predictor_area in tqdm(
            units["brain_area"].unique(), desc="predictor area", leave=False
        ):
            predictor_area_counts = counts[:, units["brain_area"] == predictor_area]
            predictor_area_counts = predictor_area_counts[
                :,
                np.random.choice(
                    predictor_area_counts.shape[1], subsample_n, replace=False
                ),
            ]

            X_train = {
                unit: basis.compute_features(
                    predictor_area_counts[:, unit].restrict(train_set)
                )
                for unit in range(subsample_n)
            }
            y_train = predicted_area_counts.restrict(train_set)
            mean_rates = np.clip(np.array(np.nanmean(y_train, axis=0)), 1e-3, None)
            init_intercept = np.log(mean_rates) * bin_size_sec

            model = nmo.glm.PopulationGLM(
                regularizer=nmo.regularizer.GroupLasso(),
                regularizer_strength=0.01,
                solver_kwargs={"max_iter": 2000}
            )
            model.fit(
                X_train,
                y_train,
                init_params=(
                    {i: np.zeros((n_basis_funcs, y_train.shape[1])) for i in X_train},
                    init_intercept,
                ),
            )

            X_test = {
                unit: basis.compute_features(
                    predictor_area_counts[:, unit].restrict(test_set)
                )
                for unit in range(subsample_n)
            }
            y_test = predicted_area_counts.restrict(test_set)
            scores = model.score(X_test, y_test)

            session_results.append(
                {
                    "dataset": dataset_num,
                    "predicted": predicted_area,
                    "predictor": predictor_area,
                    "pr2": float(scores),
                }
            )

    # Track best pair per session
    session_df = pd.DataFrame(session_results)
    mat_raw = session_df.pivot_table(
        index="predicted", columns="predictor", values="pr2", aggfunc="mean"
    ).astype(float)

    within_scores = pd.Series(np.diag(mat_raw), index=mat_raw.index)
    session_df["normalised_pr2"] = session_df["pr2"] - session_df["predicted"].map(within_scores)

    cross_area = session_df[session_df["predicted"] != session_df["predictor"]]
    best = (
        cross_area.groupby(["predicted", "predictor"])["normalised_pr2"]
        .mean()
        .reset_index()
        .sort_values("normalised_pr2", ascending=False)
    )
    best_pair = best.iloc[0]
    best_pairs.append(
        {
            "dataset": dataset_num,
            "predictor": best_pair["predictor"],
            "predicted": best_pair["predicted"],
            "normalised_pr2": best_pair["normalised_pr2"],
        }
    )

    results.append(session_results)

results = pd.DataFrame([r for session in results for r in session])
```

## Visualization

We visualize the mean pseudo-R² per brain area pair.
We normalize each row by the within-area self-prediction performance, so that the
baseline predictability of each area is factored out — values close to 0 mean the
cross-area prediction is no better than self-prediction.

```{code-cell} ipython3
mat = results.pivot_table(
    index="predicted", columns="predictor", values="pr2", aggfunc="mean"
).astype(float)

# Extract within-area scores before normalising
within_scores = pd.Series(np.diag(mat), index=mat.index)
mat_norm = mat.sub(within_scores, axis=0)

mask_diagonal = np.eye(len(mat_norm), dtype=bool)
sns.heatmap(
    mat_norm,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    mask=mask_diagonal,
    cbar_kws={"shrink": 0.7, "label": "normalised pseudo-R² (cross − within)"},
)
plt.xlabel("predictor")
plt.ylabel("predicted")
plt.tight_layout()
```

## Summary

```{code-cell} ipython3
results["normalised_pr2"] = results["pr2"] - results["predicted"].map(within_scores)

cross_area = results[results["predicted"] != results["predictor"]]
best = (
    cross_area.groupby(["predicted", "predictor"])["normalised_pr2"]
    .mean()
    .reset_index()
    .sort_values("normalised_pr2", ascending=False)
)

print("Top cross-area pairs:")
print(best.head(10).to_string(index=False))

best_pairs_df = pd.DataFrame(best_pairs)
print("\n=== Best pair per dataset ===")
print(best_pairs_df.to_string(index=False))

pair_counts = (
    best_pairs_df.groupby(["predictor", "predicted"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
print("\n=== Pair occurrence counts ===")
print(pair_counts.to_string(index=False))
```

## Answer

```{code-cell} ipython3
best_pair = best.iloc[0]
most_common = pair_counts.iloc[0]

print(
    f"Best pair (pooled across datasets): {best_pair['predictor']} → {best_pair['predicted']}"
    f"  (normalised pR²={best_pair['normalised_pr2']:.4f})"
)
print(
    f"Most common best pair across datasets: {most_common['predictor']} → {most_common['predicted']}"
    f"  ({most_common['count']}/{len(best_pairs_df)} datasets)"
)
```
