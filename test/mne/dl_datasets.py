import mne
from mne.datasets import sample
import pickle

data_path = sample.data_path()
file_paths = ['MEG/sample/sample_audvis_filt-0-40_raw.fif']

for file_path in file_paths:
    raw_fname = data_path+ "/" + file_path
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, exclude='bads')
    raw.save(file_path, tmin=0, tmax=150, picks=picks, overwrite=True)