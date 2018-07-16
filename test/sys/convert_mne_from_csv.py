import numpy as np
import mne
from mne import Epochs, pick_types, find_events

mne.set_log_level('WARNING')

def epochs_from_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1).T
    ch_types = ["eeg" for i in range(16)] + ["stim"]
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "STIM"]

    info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
    raw = mne.io.RawArray(data[1:], info)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events = mne.find_events(raw, stim_channel='STIM')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    event_id = dict(right=1, left=2)
    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
    return epochs
