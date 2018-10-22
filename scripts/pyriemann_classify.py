import timeit
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import mne
from mne import Epochs, pick_types
from mne.io import concatenate_raws
from mne.io.edf import read_raw_edf
from mne.datasets import eegbci
from mne.event import find_events
from mne.decoding import CSP

from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# from sklearn.cross_validation import cross_val_score, GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegressions
from mne_wrapper import get_raw

mne.set_log_level('WARNING')


def get_score(subject=7, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    tmin, tmax = -1., 4.

    raw = get_raw(subject, runs)
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    # cv = KFold(len(labels), 10, shuffle=True, random_state=42)
    epochs_data_train = epochs.get_data()[:, :-1]
    cov_data_train = Covariances().transform(epochs_data_train)

    ###############################################################################
    # Classification with Minimum distance to mean
    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    pl = Pipeline([("mdm", mdm)])
    params = {"mdm__metric": [dict(mean='riemann', distance='riemann')]}
    clf = GridSearchCV(pl, params, n_jobs=-1, cv=5, return_train_score=True)
    clf.fit(cov_data_train, labels)
    df = pd.DataFrame(clf.cv_results_)
    return df

    ###############################################################################
    # # Classification with Tangent Space Logistic Regression
    # clf = TSclassifier()
    # scores = cross_val_score(clf, cov_data_train, labels, cv=cv, n_jobs=1)

    # # Printing the results
    # class_balance = np.mean(labels == labels[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # ts_score = np.mean(scores)
    # print("Tangent space Classification accuracy: %f / Chance level: %f" %
    #       (ts_score, class_balance))

    ###############################################################################

    # return [subject, mdm_score, ts_score]


if __name__ == "__main__":
    df = get_score(subject=10)
    df["subject"] = pd.Series([10], index=df.index)
    print(df)
    exit()

    results = []
    for i in range(1, 110):
        print(i)
        df = get_score(subject=i, runs=[6, 10, 14],
                       event_id=dict(hands=2, feet=3))
        df["subject"] = pd.Series([i], index=df.index)
        results.append(df)
    df = pd.concat(results)
    df.to_excel(
        "data/hands_vs_feet_two_class_pyriemann_scores.xlsx", index=False)

    results = []
    for i in range(1, 110):
        print(i)
        df = get_score(subject=i, runs=[6, 10, 14],
                       event_id=dict(rest=1, hands=2, feet=3))
        df["subject"] = pd.Series([i], index=df.index)
        results.append(df)
    df = pd.concat(results)
    df.to_excel(
        "data/hands_vs_feet_three_class_pyriemann_scores.xlsx", index=False)

    results = []
    for i in range(1, 110):
        print(i)
        df = get_score(subject=i, runs=[4, 8, 12],
                       event_id=dict(left=2, right=3))
        df["subject"] = pd.Series([i], index=df.index)
        results.append(df)
    df = pd.concat(results)
    df.to_excel(
        "data/left_vs_right_two_class_pyriemann_scores.xlsx", index=False)

    results = []
    for i in range(1, 110):
        print(i)
        df = get_score(subject=i, runs=[4, 8, 12],
                       event_id=dict(rest=1, left=2, right=3))
        df["subject"] = pd.Series([i], index=df.index)
        results.append(df)
    df = pd.concat(results)
    df.to_excel(
        "data/left_vs_right_three_class_pyriemann_scores.xlsx", index=False)
