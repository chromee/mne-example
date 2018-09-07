import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mne.set_log_level('WARNING')


subject = 1
runs = [6, 10, 14]
event_id = dict(rest=1, hands=2, feet=3)
tmin, tmax = -1., 4.

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto')
             for f in raw_fnames]
print(raw_files)
raw = concatenate_raws(raw_files)
print(np.unique(raw.get_data()[-1]))

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1]

print(epochs.get_data().shape)
print(epochs_train.get_data().shape)
