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


def down_sample(x, R):
    ary = np.zeros((x.shape[0], x.shape[1], R))
    for epoch in range(x.shape[0]):
        for k in range(x.shape[1]):
            split_arr = np.linspace(0, len(x[epoch][k]), num=R+1, dtype=int)
            dwnsmpl_subarr = np.split(x[epoch][k], split_arr[1:])
            ary[epoch][k] = np.array(list(np.nanmean(item)
                                          for item in dwnsmpl_subarr[:-1]))
    return ary.reshape((ary.shape[0], ary.shape[1]*ary.shape[2]))


def get_score(subject=1):
    epochs = mne_wrapper.get_epochs(subject)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    epochs_data_train = epochs_train.get_data()
    labels = epochs.events[:, -1]

    cv = KFold(n_splits=5)
    csp = CSP(n_components=4, reg=None,
              norm_trace=False, transform_into="csp_space")
    svc = svm.SVC(kernel="linear")

    scores = []
    self_scores = []
    for train_index, test_index in cv.split(epochs_data_train):
        # fit
        x = epochs_data_train[train_index]
        y = labels[train_index]
        x = csp.fit_transform(x, y)
        x = np.fft.fft(x)
        x = np.abs(x)
        x = down_sample(x, 20)
        # print(x)
        svc.fit(x, y)
        self_scores.append(svc.score(x, y))

        # estimate
        x_test = epochs_data_train[test_index]
        y_test = labels[test_index]
        x_test = csp.transform(x_test)
        x_test = np.fft.fft(x_test)
        x_test = np.abs(x_test)
        x_test = down_sample(x_test, 20)
        score = svc.score(x_test, y_test)
        scores.append(score)

    return np.mean(self_scores), np.mean(scores)


scores = np.zeros((110, 2))
for i in range(1, 110):
    scores[i] = get_score(i)
    print(i)
print(scores.mean(axis=0))
