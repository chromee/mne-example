import numpy as np
import pandas as pd
import pickle

import mne
from mne import Epochs, pick_types
from mne.event import find_events

# pyriemann import
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances

# sklearn imports
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline

from mne_wrapper import get_raw

mne.set_log_level('WARNING')

EPOCH_COUNT = 87


def get_score(subject=7, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    tmin, tmax = -1., 4.

    # learn all suject exclude target subject. #############################
    first_sub = 2 if subject == 1 else 1
    raw = get_raw(first_sub, runs)
    for i in range(first_sub+1, 3):
        if i != subject and not (i in [88, 89, 92, 100]):
            # print(i)
            raw.append(get_raw(i, runs))
    raw.append(get_raw(subject, runs))

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)

    labels = epochs.events[:, -1]
    epochs_data_train = 1e6*epochs.get_data()[:, :-1]
    cov_data_train = Covariances().transform(epochs_data_train)

    weights = np.arange(0.1, 1.0, 0.1)
    scores = []
    for weight in weights:
        mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
        others_sample_weight_base = np.ones(
            len(epochs)-EPOCH_COUNT)*(1.-weight)
        target_sample_weight_base = np.ones(EPOCH_COUNT)*weight
        sample_weight = np.hstack(
            (others_sample_weight_base, target_sample_weight_base))

        others_size = others_sample_weight_base.size
        others_index = np.arange(others_size)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        train_scores = []
        test_scores = []
        dumy_array = np.ones(EPOCH_COUNT)
        for train_index, test_index in cv.split(dumy_array):
            train_index = np.hstack((others_index, train_index+others_size))
            x = cov_data_train[train_index]
            y = labels[train_index]
            mdm.fit(x, y, sample_weight=sample_weight[train_index])
            score = (mdm.predict(x) == y).sum()/len(train_index)
            train_scores.append(score)

            test_index = test_index + others_size
            y = mdm.predict(cov_data_train[test_index])
            score = (y == labels[test_index]).sum()/len(test_index)
            test_scores.append(score)

        train_score = np.mean(train_scores)
        test_score = np.mean(test_scores)
        scores.append([subject, weight, train_score, test_score])
        # print("train:%s test:%s" % (train_score, test_score))
    return scores


if __name__ == "__main__":
    # result = get_score(subject=7)
    # df = pd.DataFrame(result)
    # df.to_excel("data/test.xlsx", index=False)

    columns = ["subject", "target_sample_weight", "train_score", "test_score"]
    subject_count = 3

    df = pd.DataFrame(columns=columns)
    for i in range(1, subject_count):
        print("hands vs feet two class %d" % i)
        result = get_score(
            subject=i, runs=[6, 10, 14], event_id=dict(hands=2, feet=3))
        tmp_df = pd.DataFrame(result, columns=columns)
        df = df.append(tmp_df, ignore_index=True)
    df.to_excel(
        "data/transfer/hands_vs_feet_two_class_tl_pyriemann_scores.xlsx", index=False)

    ###############################################################################

    df = pd.DataFrame(columns=columns)
    for i in range(1, subject_count):
        print("hands vs feet three class %d" % i)
        result = get_score(
            subject=i, runs=[6, 10, 14], event_id=dict(rest=1, hands=2, feet=3))
        tmp_df = pd.DataFrame(result, columns=columns)
        df = df.append(tmp_df, ignore_index=True)
    df.to_excel(
        "data/transfer/hands_vs_feet_three_class_tl_pyriemann_scores.xlsx", index=False)

    ###############################################################################

    df = pd.DataFrame(columns=columns)
    for i in range(1, subject_count):
        print("left vs right two class %d" % i)
        result = get_score(
            subject=i, runs=[4, 8, 12], event_id=dict(left=2, right=3))
        tmp_df = pd.DataFrame(result, columns=columns)
        df = df.append(tmp_df, ignore_index=True)
    df.to_excel(
        "data/transfer/left_vs_right_two_class_tl_pyriemann_scores.xlsx", index=False)

    ###############################################################################

    df = pd.DataFrame(columns=columns)
    for i in range(1, subject_count):
        print("left vs right three class %d" % i)
        result = get_score(
            subject=i, runs=[4, 8, 12], event_id=dict(rest=1, left=2, right=3))
        tmp_df = pd.DataFrame(result, columns=columns)
        df = df.append(tmp_df, ignore_index=True)
    df.to_excel(
        "data/transfer/left_vs_right_three_class_tl_pyriemann_scores.xlsx", index=False)

    print("end")
