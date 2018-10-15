import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
import sklearn.pipeline
from sklearn.model_selection import KFold, cross_val_score

import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne import Epochs, pick_types, find_events
import mne_wrapper


def get_score(subject=1):
    epochs = mne_wrapper.get_epochs(subject)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    epochs_data_train = epochs_train.get_data()
    labels = epochs.events[:, -1]

    cv = KFold(n_splits=5)
    csp = CSP(n_components=4, reg=None,
              norm_trace=False, transform_into="csp_space")
    fft = mne_wrapper.FFT()
    svc = svm.SVC(kernel="linear")

    scores = []
    self_scores = []
    for train_index, test_index in cv.split(epochs_data_train):
        # fit
        x = epochs_data_train[train_index]
        y = labels[train_index]
        x = csp.fit_transform(x, y)
        x = fft.transform(x)
        # print(x)
        svc.fit(x, y)
        self_scores.append(svc.score(x, y))

        # estimate
        x_test = epochs_data_train[test_index]
        y_test = labels[test_index]
        x_test = csp.transform(x_test)
        x_test = fft.transform(x_test)
        score = svc.score(x_test, y_test)
        scores.append(score)

    return np.mean(self_scores), np.mean(scores)


print(get_score(1))

# scores = np.zeros((110, 2))
# for i in range(1, 110):
#     scores[i] = get_score(i)
#     print(i)
# print(scores.mean(axis=0))
