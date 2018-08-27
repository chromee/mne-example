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
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto')
                 for f in raw_fnames]
    raw = concatenate_raws(raw_files)

    raw.rename_channels(lambda x: x.strip('.'))

    # T3 -> T7, T4 -> T8
    ch_names = ["Fp1", "Fp2", "F7", "F3", "F4", "F8", "T7", "C3",
                "Cz", "C4", "T8", "P3", "Pz", "P4", "O1", "O2", "STI 014"]
    raw.pick_channels(ch_names)

    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True,
                       stim=False, eog=False, exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4.,
                    proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    scores = []
    epochs_data_train = epochs_train.get_data()

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP(n_components=4, reg=None,
                           log=True, norm_trace=False)

    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    self_scores = []
    scores = []
    for train, test in cv.split(epochs_data_train):
        # fit
        x = epochs_data_train[train]
        y = labels[train]
        x = csp.fit_transform(x, y)
        lda.fit(x, y)
        self_scores.append(lda.score(x, y))

        # estimate
        x_test = epochs_data_train[test]
        y_test = labels[test]
        x_test = csp.transform(x_test)
        score = lda.score(x_test, y_test)
        scores.append(score)

    # scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
    return np.mean(self_scores), np.mean(scores)


# # right vs left
# runs = [4, 8, 12]
# event_id = dict(right=2, left=3)


# hand vs feet
runs = [6, 10, 14]
event_id = dict(hands=2, feet=3)

# one subject
subject = 3
self_scores, score = get_score(subject, runs, event_id)
print("self_scores", self_scores)
print("score", score)

# # all subjects
# scores = []
# for i in range(109):
#     subject = i+1
#     score = get_score(subject, runs, event_id)
#     print(subject, score)
#     scores.append(score)
# scores = np.array(scores)
# print("average:", scores.mean())
# np.savetxt("scores.csv", scores, delimiter=",")
