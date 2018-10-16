import numpy as np
from time import sleep

import mne
from mne import Epochs
from mne.event import find_events
from mne_wrapper import get_raw
mne.set_log_level('WARNING')

from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances


def create_mdm(raw, event_id):
    tmin, tmax = -1., 4.
    events = find_events(raw, shortest_event=0, stim_channel='STI 014')
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    baseline=None, preload=True, verbose=False)
    labels = epochs.events[:, -1]
    epochs_data_train = epochs.get_data()[:, :-1]
    cov_data_train = Covariances().transform(epochs_data_train)
    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    mdm.fit(cov_data_train, labels)
    return mdm


def get_score(subject, event):
    if event == "left_vs_right":
        runs = [4, 8, 12]
        event_id = dict(rest=1, left=2, right=3)
    else:
        runs = [6, 10, 14]
        event_id = dict(rest=1, hands=2, feet=3)
    raw = get_raw(subject, runs=runs)
    data = raw.get_data()

    mdm = create_mdm(raw, event_id)

    w_length = 160  # 学習epochのlength
    w_step = 80

    current_step = 0
    count = 0
    score = 0

    print("start")
    for i in range(len(data[0])):
        current_step += 1
        if i > w_length and current_step > w_step:
            window = np.array([data[0:16, i:i+w_length]])
            X_test = Covariances().transform(window)
            # print(X_test.shape)
            label = mdm.predict(X_test)

            current_step = 0
            count += 1

            window_labels = np.array(data[16][i:i+w_length], dtype=np.int64)
            label_count = np.bincount(window_labels)
            argmax_label = np.argmax(label_count)

            print(i, label, argmax_label)

            if label == argmax_label:
                score += 1
        # sleep(1./160)
    print(subject, score/count)
    print("end")
    return score/count


get_score(1, "left_vs_right")
# scores = []
# for i in range(1, 110):
#     scores.append(get_score(i, "left_vs_right"))
#     print(i)
# np.savetxt("real_acu.csv", scores, delimiter=",")
