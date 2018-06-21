import os.path as op
import numpy as np

import mne

# data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
# raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'), preload=True)
# # raw.set_eeg_reference('average', projection=True)

data_path = op.join(mne.datasets.sample.data_path(), 'MEG', 'sample')
raw = mne.io.read_raw_fif(op.join(data_path, 'sample_audvis_raw.fif'), preload=True)

import pickle
f = open("meg_sample.pybin", "wb")
pickle.dump(raw, f)
f.close