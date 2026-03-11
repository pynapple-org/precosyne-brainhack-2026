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
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
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

dandiset_id = "218201"
np.random.seed(42)
```

## Population-level Poisson GLMs

For each dataset, we load spike trains and fit a population-level GLM for every possible pair of brain areas, including self-prediction.


### Notes

- We subsample the units to the lowest number per brain area, otherwise we might interpret performance differences caused by different numbers units as functional connectivity.
- We use NeMoS' `GroupLasso` to regularize each unit's features together.

```{code-cell} ipython3
window_size_sec = 0.8
bin_size_sec = 0.05
n_basis_funcs = 4
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=n_basis_funcs,
    window_size=int(window_size_sec // bin_size_sec),
)

results = []
for dataset_num in [8]:

    # Stream data from DANDI
    filepath = f"sub-mouse-{dataset_num}/sub-mouse-{dataset_num}_ses-None_ecephys.nwb"
    with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
        asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = lindi.LindiH5pyFile.from_hdf5_file(s3_url, local_cache=lindi.LocalCache())
    io = NWBHDF5IO(file=f)
    units = nap.NWBFile(io.read())["units"]
    units = units[units.rate>1.0]

    # Count spikes
    counts = units.count(bin_size_sec)

    # Subsample to the lowest number of units per brain area
    subsample_n = units["brain_area"].value_counts().min()

    # Define splits (we'll only look at the first 2 minutes here)
    train_set, test_set = nap.IntervalSet([0, 60]), nap.IntervalSet([60, 120])

    # Loop over all possible combinations
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

            # Fit on train set
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
            )
            model.fit(
                X_train,
                y_train,
                init_params=(
                    {
                        i: np.zeros((n_basis_funcs, y_train.shape[1]))
                        for i in X_train
                    },
                    init_intercept,
                ),
            )

            # Compute scores on test set
            X_test = {
                unit: basis.compute_features(
                    predictor_area_counts[:, unit].restrict(test_set)
                )
                for unit in range(subsample_n)
            }
            y_test = predicted_area_counts.restrict(test_set)
            scores = model.score(X_test, y_test)

            # Store results
            results.append(
                {
                    "predicted": predicted_area,
                    "predictor": predictor_area,
                    "pr2": scores,
                }
            )
```

## Visualization

We visualize the mean pseudo-R² per brain area pair to look for patterns.
We normalize each row by the performance of self-prediction such that the general predictivity of a brain area is taken out of the equation.

```{code-cell} ipython3
results = pd.DataFrame(results)
mat = weights.pivot_table(
    index="predicted", columns="predictor", values="pr2", aggfunc="mean"
).astype(float)
mat = mat.sub(np.diag(mat), axis=0)

sns.heatmap(
    mat,
    annot=True,
    cmap="RdYlGn",
    cbar_kws={"shrink": 0.7, "label": "mean pseudo-R² within - across"},
)
plt.xlabel("predictor")
plt.ylabel("predicted")
```

## Summary

```{code-cell} ipython3
cross_area = results[results["predicted"] != results["predictor"]]
best = (
    cross_area.groupby(["predicted", "predictor"])["pr2"]
    .mean()
    .reset_index()
    .sort_values("pr2", ascending=False)
)
best_pair = best.iloc[0]
```

## Answer

```{code-cell} ipython3
print(f"Brain area pair {best_pair['predictor']} → {best_pair['predicted']} "
      f"has the strongest functional connectivity (pR²={best_pair['pr2']:.4f}).")
```
