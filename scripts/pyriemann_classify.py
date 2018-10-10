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
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegressions
from mne_wrapper import get_raw

mne.set_log_level('WARNING')


def get_score(subject=7):
    ###############################################################################
    # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    runs = [6, 10, 14]
    event_id = dict(rest=1, hands=2, feet=3)
    tmin, tmax = -1., 4.

    raw = get_raw(subject, runs)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    # cross validation
    cv = KFold(len(labels), 10, shuffle=True, random_state=42)
    # get epochs
    epochs_data_train = 1e6*epochs.get_data()

    # compute covariance matrices
    cov_data_train = Covariances().transform(epochs_data_train)

    ###############################################################################
    # Classification with Minimum distance to mean
    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))

    # Use scikit-learn Pipeline with cross_val_score function
    scores = cross_val_score(mdm, cov_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    mdm_score = np.mean(scores)
    print("MDM Classification accuracy: %f / Chance level: %f" % (mdm_score,
                                                                  class_balance))

    ###############################################################################
    # Classification with Tangent Space Logistic Regression
    clf = TSclassifier()
    # Use scikit-learn Pipeline with cross_val_score function
    scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    ts_score = np.mean(scores)
    print("Tangent space Classification accuracy: %f / Chance level: %f" %
          (ts_score, class_balance))

    ###############################################################################

    return [subject, mdm_score, ts_score]


if __name__ == "__main__":
    # print(get_score(subject=7))

    columns = ["subject", "mdm_score", "ts_score"]
    scores = pd.DataFrame(columns=columns)
    for i in range(1, 110):
        print(i)
        se = pd.Series(get_score(subject=i), index=columns)
        scores = scores.append(se, ignore_index=True)
    scores.to_excel("pyriemann_scores.xlsx", index=False)
