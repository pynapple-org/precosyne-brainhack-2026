

from brainhack import config, load_lfp, load_spikes, load_trials

# set the path to the folder containing the files
# assuming files are in "root_folder/n/" where n is a number 1,...,18

root_folder = "/Users/gviejo/Pre-Cosyne-BrainHack-2026"
config.update(root_folder=root_folder)

dataset_num = 1

# load to pynapple
spikes = load_spikes(dataset_num)
lfp = load_lfp(dataset_num)
trials = load_trials(dataset_num)

