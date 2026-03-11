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
import lindi


# anonymized dataset
dandiset_id = "218201"
filepath = "sub-mouse-15/sub-mouse-15_ses-None_ecephys.nwb"

with DandiAPIClient(api_url="https://api.sandbox.dandiarchive.org/api") as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)


f = lindi.LindiH5pyFile.from_hdf5_file(
    s3_url,  #"https://api.sandbox.dandiarchive.org/api/assets/9aad5ba1-9906-46a5-8f6f-611978dfa2e6/download/",
    local_cache=lindi.LocalCache()
)
io = NWBHDF5IO(file=f)

data = nap.NWBFile(io.read())

spikes = data["units"]

