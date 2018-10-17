import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

EXCLUDE_SUBJECTS = [88, 89, 92, 100]
TRANS_SUBJECT_COUNT = 110
# EXCLUDE_SUBJECTS = [2, 3, 5]
# TRANS_SUBJECT_COUNT = 9


def get_score(subject=7, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    if subject in EXCLUDE_SUBJECTS:
        return
    tmin, tmax = -1., 4.

    scores_each_wright = []
    weights = np.arange(0.1, 1.0, 0.1)
    for weight in weights:
        first_sub = 2 if subject == 1 else 1
        raw = get_raw(subject, runs)
        scores_each_wright.append([])
        for i in range(first_sub, TRANS_SUBJECT_COUNT):
            print(i)
            if i == subject or (i in EXCLUDE_SUBJECTS):
                continue
            raw.append(get_raw(i, runs))

            events = find_events(raw, shortest_event=0, stim_channel='STI 014')
            epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                            baseline=None, preload=True, verbose=False)
            labels = epochs.events[:, -1]
            epochs_data_train = 1e6*epochs.get_data()[:, :-1]
            cov_data_train = Covariances().transform(epochs_data_train)

            target_sample_weight_base = np.ones(EPOCH_COUNT)*weight
            others_sample_weight_base = np.ones(
                len(epochs)-EPOCH_COUNT)*(1.-weight)
            sample_weight = np.hstack(
                (target_sample_weight_base, others_sample_weight_base))

            others_size = others_sample_weight_base.size
            others_index = np.arange(others_size)

            mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            train_scores = []
            test_scores = []
            dumy_array = np.ones(EPOCH_COUNT)

            for train_index, test_index in cv.split(dumy_array):
                train_index = np.hstack(
                    (others_index, train_index+others_size))
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
            scores_each_wright[(int)(weight*10)-1].append(
                [subject, i, train_score, test_score])

        y = np.array([x[-1] for x in scores_each_wright[-1]])
        x = np.arange(1, y.size+1)
        print(subject, weight)
        plt.subplot(3, 3, (int)(weight*10))
        plt.plot(x, y)
    plt.savefig("data/transfer/gradually/subject_%d.svg" % (subject))
    return scores_each_wright


if __name__ == "__main__":
    subject = 7
    result = get_score(subject=subject)
    df = pd.DataFrame(result)
    df.to_excel("data/test_subject_%d.xlsx" % subject, index=False)
