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

# Question 4: During which trial segment is variable C best decoded?

Another way to frame this question is: during which trial segment is variable C best predicted from neural activity? To answer this version of the question, we'll use a classification GLM, which is useful for predicting the labels of categorical variables like variable C. This model avilable in NeMoS as [`ClassifierGLM`](https://nemos.readthedocs.io/en/latest/generated/glm/nemos.glm.ClassifierGLM.html#nemos.glm.ClassifierGLM). 

We need to compare decoding during three trial segments: from trial start to simulus onset, from stimulus onset to outcome time, and from outcome time to trial end. 

```{code-cell} ipython3
# imports
%matplotlib inline
import warnings

from pynwb import NWBHDF5IO

from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py

import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nemos as nmo
import jax
import scipy.stats as stats

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
```

First, we'll stream a representative animal directly from DANDI.

```{code-cell} ipython3
# anonymized dataset
dandiset_id = "218201"
filepath = "sub-mouse-13/sub-mouse-13_ses-None_ecephys.nwb"

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

We can load the streamed file directly with Pynapple. We'll grab the relevant data: the spike times and trial information. Additionally, we'll exclude any trials where the four demarcating time points don't occur consecutively.

```{code-cell} ipython3
# load in the streamed file
data = nap.NWBFile(io.read())
print(data)

# get spike times and trial data
spikes = data["units"]
trials = data["trials"]

# remove trials with bad timing
trials = trials[
    (trials.start < trials.stim_start) & 
    (trials.stim_start < trials.outcome) &
    (trials.outcome < trials.end)
]

# get number of classes for ClassifierGLM
n_classes = len(np.unique(trials.variable_C))
```

Let's define the three segments each as a pynapple [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html)

```{code-cell} ipython3
# define interval sets for the three segments of interest
start_to_stim = nap.IntervalSet(start=trials.start, end=trials.stim_start, metadata=trials.metadata)
stim_to_outcome = nap.IntervalSet(start=trials.stim_start, end=trials.outcome,  metadata=trials.metadata)
outcome_to_end = nap.IntervalSet(start=trials.outcome, end=trials.end, metadata=trials.metadata)
```

## Simple cross-validation

We will want to use cross-validation to determine how well the neural activity predicts variable C. In other words, we'll train the model on one subset of the data and calculate how well it predicts the labels of a separate, held-out subset of the data. We can do this simply by splitting the data in two halves. We'll split the data into training and testing sets by taking every other trial; this way, we have even sampling throughout the session in both sets. Additionally, we'll split evenly within each category of C in order to preserve the statistics of each category in both subsets.

We'll fit the `ClassifierGLM` on the train set, and then predict the label of Variable C on the test set. Using the predicted label, we can compute the accuracy of how well it matches the true labels in the test set.

```{code-cell} ipython3
# define interval sets for the three segments of interest
start_to_stim = nap.IntervalSet(start=trials.start, end=trials.stim_start, metadata=trials.metadata)
stim_to_outcome = nap.IntervalSet(start=trials.stim_start, end=trials.outcome,  metadata=trials.metadata)
outcome_to_end = nap.IntervalSet(start=trials.outcome, end=trials.end, metadata=trials.metadata)

seg_accuracy = np.zeros(3)
for s,seg in enumerate([start_to_stim, stim_to_outcome, outcome_to_end]):
    # use half of trials for train set, with even split within category
    seg_train = seg.groupby_apply("variable_C", lambda x: x[::2].as_dataframe())
    # concatenate results and sort
    seg_train = pd.concat(seg_train.values()).sort_values("start").reset_index(drop=True)
    # save in intervalset
    seg_train = nap.IntervalSet(seg_train)
    # use set_diff to get remaining intervals for test set
    seg_test = seg.set_diff(seg_train)
    
    # get spike counts and variable_C in relevant segment during train trials
    counts_train = spikes.restrict(seg_train).count()
    y_train = seg_train.variable_C
    
    # define a nemos ClassifierGLM model and fit to training data
    model = nmo.glm.ClassifierGLM(n_classes=n_classes, solver_kwargs={"maxiter": 1000})
    model.fit(counts_train, y_train)

    # get spike counts and variable_C for test trials
    counts_test = spikes.restrict(seg_test).count()
    y_test = seg_test.variable_C

    # get predicted labels and compute accuracy
    y_pred = model.predict(counts_test)
    seg_accuracy[s] = np.mean(y_pred == y_test)

print("Prediction accuracy:")
print(" Trial start --> Stim start:", seg_accuracy[0])
print(" Stim start --> Outcome:    ", seg_accuracy[1])
print(" Outcome --> Trial end:     ", seg_accuracy[2])
```

Based on these results, it appears that prediction accuracy is best in the second segment from stimulus onset to outcome time.

## Better cross-validation with sklearn

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold


y = trials.variable_C

n_splits = 6
skf = StratifiedKFold(n_splits=n_splits)
model = nmo.glm.ClassifierGLM(n_classes=n_classes, solver_kwargs={"maxiter": 1000})
seg_accuracy_kfold = np.zeros((n_splits,3))

for s,seg in enumerate([start_to_stim, stim_to_outcome, outcome_to_end]):
    counts = spikes.restrict(seg).count()
    for i, (train_index, test_index) in enumerate(skf.split(np.zeros_like(y), y)):
        y_train, y_test = y[train_index], y[test_index]
        counts_train, counts_test = counts[train_index,:], counts[test_index,:]

        model.fit(counts_train,y_train)
        y_pred = model.predict(counts_test)
        seg_accuracy_kfold[i,s] = np.mean(y_pred == y_test)
    
```

```{code-cell} ipython3
print(stats.wilcoxon(seg_accuracy_kfold[:,0],seg_accuracy_kfold[:,1]))
print(stats.wilcoxon(seg_accuracy_kfold[:,0],seg_accuracy_kfold[:,2]))
print(stats.wilcoxon(seg_accuracy_kfold[:,1],seg_accuracy_kfold[:,2]))
```

```{code-cell} ipython3
res = stats.bootstrap((seg_accuracy_kfold,), np.mean, confidence_level=0.95, n_resamples=10000, method="percentile", axis=0)
acc_means = np.mean(seg_accuracy_kfold, axis=0)
plt.bar(["Trial start --> Stim start", "Stim start --> Outcome", "Outcome --> Trial end"], acc_means, yerr=[acc_means-res.confidence_interval.low, res.confidence_interval.high-acc_means])
plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Average decoding accuracy across folds")
plt.plot([0, 1], [0.91,0.91], 'k')
plt.text(0.43, 0.91, "*")
plt.plot([0, 2], [0.86,0.86], 'k')
plt.text(0.93, 0.86, "*")
plt.plot([1, 2], [0.81,0.81], 'k')
plt.text(1.47, 0.82, "n.s.")
```

```{code-cell} ipython3

```
