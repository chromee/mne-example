"""
====================================================================
Motor imagery classification
====================================================================

Classify Motor imagery data with Riemannian Geometry.
"""
# generic import
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# mne import
import mne
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegressions
from mne_wrapper import get_raw
import copy

mne.set_log_level('WARNING')


def get_score(subject=7):
    runs = [6, 10, 14]
    event_id = dict(rest=1, hands=2, feet=3)
    tmin, tmax = -1., 4.

    # learn all suject exclude target subject. #############################
    raw = get_raw(2, runs)
    for i in range(3, 6):
        if i != subject and not (i in [88, 89, 92, 100]):
            print(i)
            raw.append(get_raw(i, runs))

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]
    epochs_data_train = 1e6*epochs.get_data()
    cov_data_train = Covariances().transform(epochs_data_train)

    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    mdm.fit(cov_data_train, labels)

    ###########################################################################

    raw = get_raw(subject, runs)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]
    epochs_data_train = 1e6*epochs.get_data()
    cov_data_train = Covariances().transform(epochs_data_train)

    # for sample_weight in np.arange(1, 3, 0.5):
    #     mdm_cp = copy.copy(mdm)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for train, test in cv.split(labels):
        sample_weight = np.ones(train.size)*0.1
        mdm.fit(cov_data_train[train], labels[train],
                sample_weight=sample_weight)
        y = mdm.predict(cov_data_train[test])
        score = (y == labels[test]).sum()/len(test)
        scores.append(score)
    # scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    mdm_score = np.mean(scores)
    print("MDM Classification accuracy: %f / Chance level: %f" % (mdm_score,
                                                                  class_balance))

    # ###############################################################################
    # # Classification with Tangent Space Logistic Regression
    # clf = TSclassifier()
    # # Use scikit-learn Pipeline with cross_val_score function
    # scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(labels == labels[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # ts_score = np.mean(scores)
    # print("Tangent space Classification accuracy: %f / Chance level: %f" %
    #       (ts_score, class_balance))

    # ###############################################################################

    # return [subject, mdm_score]


if __name__ == "__main__":
    print(get_score(subject=7))

    # columns = ["subject", "mdm_score", "ts_score"]
    # scores = pd.DataFrame(columns=columns)
    # for i in range(1, 110):
    #     print(i)
    #     se = pd.Series(get_score(subject=i), index=columns)
    #     scores = scores.append(se, ignore_index=True)
    # scores.to_excel("pyriemann_scores.xlsx", index=False)
