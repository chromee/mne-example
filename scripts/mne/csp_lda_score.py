import os, sys, numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.mne_wrapper import get_epochs

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

import mne.decoding
from statistics import mean

mne.set_log_level('WARNING')


def get_score(subject, runs, event_id):
    epochs = get_epochs(subject, runs, event_id)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    scores = []
    epochs_data_train = epochs_train.get_data()

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP(n_components=4, reg=None,
                           log=True, norm_trace=False)
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
