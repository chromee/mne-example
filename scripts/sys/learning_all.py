import pickle
import numpy as np
from pathlib import Path
from datetime import datetime as dt

import sklearn.discriminant_analysis
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events
import convert_mne_from_csv

lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)

def join_epochs(path, data):
    load_data = np.loadtxt(path, delimiter=",", skiprows=1).T
    if data is None:
        data = load_data
    else:
        data = np.c_[data, load_data]
    print(len(data[0]))
    return data

root = Path("./data/csv")
data = None
for path in root.iterdir():
    if path.is_file():
        tmp = join_epochs(path, data)
        data = tmp

ch_types = ["eeg" for i in range(16)] + ["stim"]
ch_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", "O2", "STIM"]

info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
raw = mne.io.RawArray(data[1:], info)
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events = mne.find_events(raw, stim_channel='STIM')
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

event_id = dict(right=1, left=2)
epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
epochs_data = epochs.get_data()
labels = epochs.events[:, -1] - 1

X_train = csp.fit_transform(epochs_data, labels)
lda.fit(X_train, labels)

file_name = "data/models/csp_eeg_python36_all.pickle"
with open(file_name, mode='wb') as csp_file:
    pickle.dump(csp, csp_file)

file_name = "data/models/lda_eeg_python36_all.pickle"
with open(file_name, mode='wb') as lda_file:
    pickle.dump(lda, lda_file)

########################################

def load_model(path):
    epochs = convert_mne_from_csv.epochs_from_csv(path)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    csp_path = "data/models/csp_eeg_python36_all.pickle"
    with open(csp_path, 'rb') as pickle_file:
        csp = pickle.load(pickle_file)
    lda_path = "data/models/lda_eeg_python36_all.pickle"
    with open(lda_path, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)

    X_test = csp.transform(epochs_data)
    score = lda.score(X_test, labels)
    print(path, score)

root = Path("./data/csv")
for path in root.iterdir():
    if path.is_file():
        load_model(path)

