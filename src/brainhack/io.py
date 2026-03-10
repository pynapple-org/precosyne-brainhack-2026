import re
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

from brainhack.config import config


def _get_base_path(dataset_num: int, root_folder: str | Path | None = None) -> Path:
    if root_folder is None:
        root_folder = config.root_folder
    return Path(root_folder) / f"{dataset_num}"


def load_trials(
    dataset_num: int, root_folder: str | Path | None = None
) -> nap.IntervalSet:
    base_path = _get_base_path(dataset_num, root_folder)
    trials = pd.read_csv(base_path / "trial_data.csv", index_col=0)
    meta = trials.copy().drop(columns=["trial_start", "trial_end"])
    trials = nap.IntervalSet(
        trials["trial_start"].to_numpy(), trials["trial_end"].to_numpy(), metadata=meta
    )
    return trials


def load_spikes(
    dataset_num: int,
    root_folder: str | Path | None = None,
    load_waveforms: bool = False,
    lfps: nap.TsdFrame | None = None,
) -> nap.TsGroup:
    """Load spike times into a TsGroup with metadata."""
    base_path = _get_base_path(dataset_num, root_folder)
    spike_times = np.load(base_path / "spikes.npy")
    clusters = np.load(base_path / "clusters.npy")
    brain_area = np.load(base_path / "brain_area.npy", allow_pickle=True).item()
    if load_waveforms:
        kwargs = dict(waveforms=np.load(base_path / "waveforms.npy"))
    else:
        kwargs = {}
    if lfps is not None:
        ep = lfps.time_support
    else:
        ep = nap.IntervalSet(0, max(spike_times))
    spikes = nap.Tsd(
        spike_times,
        clusters,
        time_support=ep
    ).to_tsgroup()
    spikes.set_info(**brain_area, **kwargs)
    # Safety check that brain area order and unit order is matching
    # should be ordered already...
    if np.any(spikes.cluster_id != spikes.index):
        raise ValueError(
            "Fix load_spikes by sorting `brain_area['cluster_id']` to match `spike.index`"
        )
    return spikes


def load_lfp(
    dataset_num: int,
    root_folder: str | Path | None = None,
    fs_hz=500,
    electrode_spacing_um=20,
) -> nap.TsdFrame:
    base_path = _get_base_path(dataset_num, root_folder)
    lfps_per_area = [
        np.load(file_path) for file_path in sorted(base_path.glob("*lfp*.npy"))
    ]
    # this is sorted, so should be 1,2,3 but to be sure:
    brain_area = [
        int(re.search(r"lfp_(\d+)\.npy$", file_path.as_posix()).group(1))
        for file_path in sorted(base_path.glob("*lfp*.npy"))
    ]
    brain_area = np.concatenate(
        [i * np.ones(l.shape[0]) for i, l in zip(brain_area, lfps_per_area)]
    )
    # referenced to first
    depth_um = np.concatenate(
        [electrode_spacing_um * np.arange(l.shape[0]) for l in lfps_per_area]
    )
    time = 1 / fs_hz * np.arange(lfps_per_area[0].shape[1])
    lfps = np.concatenate(lfps_per_area, axis=0).T
    lfps = nap.TsdFrame(
        time, lfps, metadata=dict(depth_um=depth_um, brain_area=brain_area)
    )
    return lfps
