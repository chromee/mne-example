import numpy as np
import matplotlib.pyplot as plt

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import mne.decoding

mne.set_log_level('WARNING')

##########  Raw ############

data = np.random.randn(5, 1000)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)

custom_raw = mne.io.RawArray(data, info)

custom_raw.plot()

#####################  Epochs  ##########################

# Generate some random data: 10 epochs, 5 channels, 2 seconds per epoch
sfreq = 100
data = np.random.randn(10, 5, sfreq * 2)

# Initialize an info structure
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=sfreq
)

# Create an event matrix: 10 events with alternating event codes
events = np.array([
    [0, 0, 1],
    [1, 0, 2],
    [2, 0, 1],
    [3, 0, 2],
    [4, 0, 1],
    [5, 0, 2],
    [6, 0, 1],
    [7, 0, 2],
    [8, 0, 1],
    [9, 0, 2],
])


event_id = dict(smiling=1, frowning=2)
tmin = -0.1
custom_epochs = mne.EpochsArray(data, info, events, tmin, event_id)

print(custom_epochs)

# We can treat the epochs object as we would any other
custom_epochs['smiling'].average().plot(time_unit='s')

############  Evoked  ###############

# # The averaged data
# data_evoked = data.mean(0)

# # The number of epochs that were averaged
# nave = data.shape[0]

# # A comment to describe to evoked (usually the condition name)
# comment = "Smiley faces"

# # Create the Evoked object
# evoked_array = mne.EvokedArray(data_evoked, info, tmin, comment=comment, nave=nave)
# print(evoked_array)
# evoked_array.plot(time_unit='s')

