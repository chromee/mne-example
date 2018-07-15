import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dt

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf

mne.set_log_level('WARNING')

def culc_by_csp_and_lda(path, csp_path, lda_path):
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
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    with open(csp_path, 'rb') as pickle_file:
        csp = pickle.load(pickle_file)
    with open(lda_path, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)

    X_test = csp.transform(epochs_data)
    score = lda.score(X_test, labels)
    print(score)

# root = Path("./data/csv")
# for path in root.iterdir():
#     if path.is_file():
#         culc_by_csp_and_lda(path)

path = Path("./data/csv/jtanaka_MIK_14_05_2016_13_33_15_0000.csv")
cap_path = "./data/models/csp_eeg_python36_jtanaka_MIK_14_05_2016_13_33_15_0000.csv_20180715.pickle"
lda_path = "./data/models/lda_eeg_python36_jtanaka_MIK_14_05_2016_13_33_15_0000.csv_20180715.pickle"
culc_by_csp_and_lda(path, cap_path, lda_path)