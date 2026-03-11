# precosyne-brainhack-2026

Curated answers for the [CON²PHYS hackathon 2026](https://pre-cosyne-brainhack.github.io/hackathon2026/posts/con2phys/), which investigates how methodological choices impact scientific conclusions when analyzing identical electrophysiology data.

```{toctree}
:maxdepth: 1
:caption: Questions

Question 1 <question1>
Question 2 <question2>
Question 4 <question4>
```

## Installation

### 1. Install pynapple and NeMoS

```bash
pip install pynapple nemos
```

### 2. Install this repo

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/pre-cosyne-brainhack/precosyne-brainhack-2026.git
cd precosyne-brainhack-2026
pip install -e .
```

## Loading the data

Set the path to the folder containing the dataset files (structured as `root_folder/n/` where `n` is 1–18) and use the provided helper functions:

```python
from brainhack import config, load_lfp, load_spikes, load_trials

root_folder = "/path/to/Pre-Cosyne-BrainHack-2026"
config.update(root_folder=root_folder)

dataset_num = 1  # choose a dataset (1 to 18)

spikes = load_spikes(dataset_num)
lfp = load_lfp(dataset_num)
trials = load_trials(dataset_num)
```

See `example_load.py` for a full example.

