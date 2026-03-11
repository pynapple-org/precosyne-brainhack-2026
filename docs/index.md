---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
orphan: true
---

# PreCosyne BrainHack 2026

```{code-cell} ipython3
%matplotlib inline
%load_ext autoreload
%autoreload 2
import warnings

from pynwb import NWBHDF5IO

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py


# anonymized dataset
dandiset_id = "218201"
filepath = "sub-mouse-15/sub-mouse-15_ses-None_ecephys.nwb"


with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# first, create a virtual filesystem based on the http protocol
fs = fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = NWBHDF5IO(file=file, load_namespaces=True)
io
```

```{code-cell} ipython3
import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nemos as nmo

data = nap.NWBFile(io.read())
data
```

```{code-cell} ipython3
spikes = data["units"]
trials = data["trials"]
```

```{code-cell} ipython3
# remove bad timing
trials = trials[
    (trials.start < trials.stim_start) & 
    (trials.stim_start < trials.outcome) &
    (trials.outcome < trials.end)
]
n_classes = len(np.unique(trials.variable_C))
```

```{code-cell} ipython3
seg1 = nap.IntervalSet(start=trials.start, end=trials.stim_start, metadata=trials.metadata)
seg2 = nap.IntervalSet(start=trials.stim_start, end=trials.outcome,  metadata=trials.metadata)
seg3 = nap.IntervalSet(start=trials.outcome, end=trials.end, metadata=trials.metadata)

seg_acc = np.zeros(3)
for s,seg in enumerate([seg1, seg2, seg3]):
    # even split within category
    seg_train = seg.groupby_apply("variable_C", lambda x: x[::2].as_dataframe())
    seg_train = nap.IntervalSet(pd.concat(seg_train.values()).sort_values("start").reset_index(drop=True))
    seg_test = seg.set_diff(seg_train)
    counts_train = spikes.restrict(seg_train).count()
    y_train = seg_train.variable_C
    model = nmo.glm.ClassifierGLM(n_classes=n_classes, solver_kwargs={"maxiter": 1000})
    model.fit(counts_train, y_train)
    counts_test = spikes.restrict(seg_test).count()
    y_test = seg_test.variable_C
    y_pred = model.predict(counts_test)
    seg_acc[s] = np.mean(y_pred == y_test)
```

```{code-cell} ipython3
print(seg_acc)
```
