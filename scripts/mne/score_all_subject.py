import numpy as np

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import mne.decoding

from statistics import mean

mne.set_log_level('WARNING')

def get_score(subject, runs, event_id):
    raw_fnames = eegbci.load_data(subject, runs)
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
    raw = concatenate_raws(raw_files)

    raw.rename_channels(lambda x: x.strip('.'))

    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    scores = []
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)

    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    return np.mean(scores)

scores = []
for i in range(109):
    subject = i+1

    # right vs left
    runs = [4, 8, 12]
    event_id = dict(right=2, left=3)

    # # hand vs feet
    # runs = [6, 10, 14]
    # event_id = dict(hands=2, feet=3)

    score = get_score(subject, runs, event_id)
    print(subject, score)
    scores.append(score)

scores = np.array(scores)
print("average:", scores.mean())
np.savetxt("scores.csv", scores, delimiter=",")