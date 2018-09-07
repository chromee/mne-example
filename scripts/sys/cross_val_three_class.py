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


def culc_by_csp_and_lda(subject, runs=[6, 10, 14], event_id=dict(rest=1, hands=2, feet=3)):
    tmin, tmax = -1., 4.

    raw = get_raw(subject, runs=[6, 10, 14],
                  event_id=dict(rest=1, hands=2, feet=3))
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

    # lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda = svm.SVC()
    csp = mne.decoding.CSP(n_components=4, reg=None,
                           log=True, norm_trace=False)

    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('SVM', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    return np.mean(scores)


score = culc_by_csp_and_lda(1)
print(score)

for i in range(1, 10):
    score = culc_by_csp_and_lda(i)
    print(score)
