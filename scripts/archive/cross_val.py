import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events

from mne_wrapper import get_raw

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
    labels = epochs.events[:, -1] - 1

    epochs_data_train = epochs_train.get_data()

    cv = KFold(n_splits=5)

    svc = svm.SVC(kernel="linear")
    csp = mne.decoding.CSP(n_components=n_components, reg=None,
                           log=True, norm_trace=False)

    scores = []
    self_scores = []
    for train, test in cv.split(epochs_data_train):
        # fit
        x = epochs_data_train[train]
        y = labels[train]
        x = csp.fit_transform(x, y)
        svc.fit(x, y)
        self_scores.append(svc.score(x, y))

        # estimate
        x_test = epochs_data_train[test]
        y_test = labels[test]
        x_test = csp.transform(x_test)
        score = svc.score(x_test, y_test)
        scores.append(score)
    return np.mean(self_scores), np.mean(scores)


print(get_cross_val_score(1))


def get_scores_self_scores(runs, event_id):
    scores = []
    self_scores = []
    for i in range(1, 110):
        self_score, score = get_cross_val_score(
            i, runs=runs, event_id=event_id)
        scores.append(score)
        self_scores.append(self_score)
        # print(score)
        # print(self_score)
    scores = np.array(scores)
    self_scores = np.array(self_scores)
    return scores, self_scores


results = []

# runs = [4, 8, 12]
# event_id = dict(left=2, right=3)
# scores, self_scores = get_scores_self_scores(runs, event_id)
# results.append(scores)
# results.append(self_scores)

# runs = [6, 10, 14]
# event_id = dict(hands=2, feet=3)
# scores, self_scores = get_scores_self_scores(runs, event_id)
# results.append(scores)
# results.append(self_scores)


# runs = [4, 8, 12]
# event_id = dict(rest=1, left=2, right=3)
# scores, self_scores = get_scores_self_scores(runs, event_id)
# results.append(scores)
# results.append(self_scores)


# runs = [6, 10, 14]
# event_id = dict(rest=1, hands=2, feet=3)
# scores, self_scores = get_scores_self_scores(runs, event_id)
# results.append(scores)
# results.append(self_scores)

# results = np.array(results)
# np.savetxt("svm_acu.csv", results.T, delimiter=",")
