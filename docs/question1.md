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

# Question 1: Ripple Density per Brain Area

**Which brain area (if any) has the highest density of ripples (i.e. "hippocampal" ripples traditionally occurring during sharp wave-ripples)?**

## Approach

We detect ripple events in the LFP signal for each brain area by bandpass filtering (80–150 Hz), computing the Hilbert amplitude envelope, z-scoring, and thresholding. We demonstrate the full pipeline on dataset 15.

## Setup

```{code-cell} ipython3
%matplotlib inline
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import lindi
import pynapple as nap
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import hilbert, windows

dandiset_id = "218201"
dataset_num = 15
```

## Load LFP

```{code-cell} ipython3
filepath = f"sub-mouse-{dataset_num}/sub-mouse-{dataset_num}_ses-None_ecephys.nwb"
with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
f = lindi.LindiH5pyFile.from_hdf5_file(s3_url, local_cache=lindi.LocalCache())
io = NWBHDF5IO(file=f)
data = nap.NWBFile(io.read())
print(data)
```

We extract the LFP data for each brain area, restricting to the first 60 seconds for computational efficiency.
```{code-cell} ipython3

lfps = {
    1: data['lfp_area_1'].get(0, 60),
    2: data['lfp_area_2'].get(0, 60),
    3: data['lfp_area_3'].get(0, 60)
}

print(lfps[1])
```

## Power spectral density per brain area

We compute the power spectral density and identify the 3 channels with the highest ripple-band power (120–200 Hz) per area to reduce computational load.

```{code-cell} ipython3
powers = {}
ripple_powers = {}
for area in [1, 2, 3]:
    powers[area] = nap.compute_mean_power_spectral_density(lfps[area], 10, fs=500, ep=nap.IntervalSet(0, 60))
    ripple_powers[area] = powers[area].loc[120:200].sum(0)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4))
for area in [1, 2, 3]:
    ax.semilogy(powers[area].index, powers[area].mean(1), label=f"Area {area}")
ax.axvspan(120, 200, alpha=0.15, color="red", label="Ripple band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_title("Power spectral density per brain area")
ax.legend()
plt.tight_layout()
plt.show()
```

## Select best channels and bandpass filter

We keep the 3 channels with highest ripple-band power per area and apply a bandpass filter (120–200 Hz).

```{code-cell} ipython3
flfps = {}
for area in [1, 2, 3]:
    best_channels = ripple_powers[area].sort_values().index[-3:].values
    flfps[area] = nap.apply_bandpass_filter(lfps[area][:, best_channels], cutoff=(120, 200), fs=500)
```

```{code-cell} ipython3
ep = nap.IntervalSet(32, 34)

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i, area in enumerate([1, 2, 3]):
    axes[i].plot(lfps[area].restrict(ep).t, lfps[area].restrict(ep)[:, 0], label="Raw", alpha=0.7)
    axes[i].plot(flfps[area].restrict(ep).t, flfps[area].restrict(ep)[:, 0], label="Filtered (120-200 Hz)", alpha=0.9)
    axes[i].set_ylabel(f"Area {area}")
    if i == 0:
        axes[i].legend(loc="upper right")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Raw vs bandpass-filtered LFP (2 s excerpt)")
plt.tight_layout()
plt.show()
```

## Hilbert transform — amplitude envelope

We compute the Hilbert transform of the filtered LFP to extract the amplitude envelope, which reflects ripple strength over time. 
We plot the envelope alongside the filtered signal for visual confirmation.

```{code-cell} ipython3
envelopes = {}
for area in [1, 2, 3]:
    analytic_signal = hilbert(flfps[area].values, axis=0)
    envelopes[area] = nap.TsdFrame(
        t=flfps[area].t, d=np.abs(analytic_signal), columns=flfps[area].columns
    )
```

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for i, area in enumerate([1, 2, 3]):
    axes[i].plot(flfps[area][:,0].restrict(ep), alpha=0.5, label="Filtered")
    axes[i].plot(envelopes[area][:,0].restrict(ep), color="C3", label="Envelope")
    axes[i].set_ylabel(f"Area {area}")
    if i == 0:
        axes[i].legend(loc="upper right")
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Filtered LFP and Hilbert amplitude envelope")
plt.tight_layout()
plt.show()
```

## Smooth and z-score

We smooth the envelope with a moving average and z-score across time, then average across channels within each area.

```{code-cell} ipython3
nSS = {}  # Normalized Smoothed Signal
for area in [1, 2, 3]:
    smoothed = envelopes[area].convolve(np.ones(7) / 7)
    z = (smoothed - smoothed.mean(0)) / smoothed.std(0)
    nSS[area] = z.mean(1)  # average across channels -> Tsd

area = 1
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes[0].plot(nSS[area].restrict(ep), color="C0")
axes[0].axhline(3, color="red", linestyle="--", label="Threshold (3 SD)")
axes[0].set_title(f"Area {area} — Normalized Smoothed Signal (nSS)")
axes[0].legend()
axes[1].plot(envelopes[area].restrict(ep), color="C0", alpha=0.3, label="Envelope")
axes[1].set_title(f"Area {area} — Smoothed Envelope")
axes[2].plot(flfps[area].restrict(ep)[:,0], color="C0", alpha=0.5, label="Filtered LFP")
axes[2].set_title(f"Area {area} — Filtered LFP")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.show()
```

## Ripple detection

We detect ripple events by thresholding the nSS signal above 3 standard deviations
We further filter detected events to keep only those between 30 ms and 300 ms in duration, which are typical for hippocampal ripples. 
Finally, we plot the detected ripple events on top of the filtered LFP signal for visual confirmation.

```{code-cell} ipython3
ripples = {}
for area in [1, 2, 3]:
    ripple_events = nSS[area].threshold(3, method="above")
    ripple_ep = ripple_events.time_support
    ripple_ep = ripple_ep.drop_short_intervals(0.03, time_units="s")
    ripple_ep = ripple_ep.drop_long_intervals(0.3, time_units="s")    
    ripples[area] = ripple_ep
```


```{code-cell} ipython3
# Plot detected ripples on filtered LFP
fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
for i, area in enumerate([1, 2, 3]):
    axes[i].plot(flfps[area].restrict(ep)[:,0], color=f"C{i}", linewidth=1)
    for _, row in ripples[area].as_dataframe().iterrows():
        axes[i].axvspan(row["start"], row["end"], alpha=0.3, color="red")
    axes[i].set_ylabel(f"Area {area} (z)")
    axes[i].set_title(f"Area {area}", fontsize=9)
    axes[i].set_xlim(ep.start[0], ep.end[0])
axes[-1].set_xlabel("Time (s)")
fig.suptitle("Detected ripple events (red shading)")
plt.tight_layout()
plt.show()
```

## Ripple density

```{code-cell} ipython3
recording_duration = lfps[1].time_support.tot_length("s")
for area in [1, 2, 3]:
    density = len(ripples[area]) / recording_duration
    print(f"Area {area}: {len(ripples[area])} ripples — {density:.4f} ripples/s")
```

## Answer

```{code-cell} ipython3
densities = {area: len(ripples[area]) / recording_duration for area in [1, 2, 3]}
best_area = max(densities, key=densities.get)
print(f"Brain area {best_area} has the highest ripple density ({densities[best_area]:.4f} ripples/s) in dataset {dataset_num}.")
```