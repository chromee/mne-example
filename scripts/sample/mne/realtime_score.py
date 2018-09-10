import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_raw

mne.set_log_level('WARNING')


def get_cross_val_score(subject, runs=[6, 10, 14], event_id=dict(rest=1, hands=2, feet=3), n_components=4):
    tmin, tmax = -1., 4.
    raw = get_raw(subject, runs)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1]

    scores = []
    epochs_data_train = epochs_train.get_data()

    cv = KFold(n_splits=5)

    svc = svm.SVC()
    csp = mne.decoding.CSP(n_components=n_components, reg=None,
                           log=True, norm_trace=False)

    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('SVM', svc)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    return np.mean(scores)
