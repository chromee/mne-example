import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import mne
from mne_wrapper import get_raw


def get_score(subject, event):
    if event == "left_vs_right":
        runs = [4, 8, 12]
    else:
        runs = [6, 10, 14]

    raw = get_raw(subject, runs=runs)
    data = raw.get_data()

    with open("./data/models/three/" + event + "/csp/csp_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
        csp = pickle.load(pickle_file)
    with open("./data/models/three/" + event + "/svm/svm_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
        svm = pickle.load(pickle_file)

    w_length = 160  # 学習epochのlength
    w_step = 80

    current_step = 0
    count = 0
    score = 0

    for i in range(len(data[0])):
        current_step += 1
        if i > w_length and current_step > w_step:
            window = np.array([data[0:16, i:i+w_length]])
            X_test = csp.transform(window)
            label = svm.predict(X_test)

            current_step = 0
            count += 1

            window_labels = np.array(data[16][i:i+w_length], dtype=np.int64)
            label_count = np.bincount(window_labels)
            argmax_label = np.argmax(label_count)

            # print(i, label, argmax_label)

            if label == argmax_label:
                score += 1
        # sleep(interval)
    # print(i, score/count)
    return score/count


scores = []
for i in range(1, 110):
    scores.append(get_score(i, "left_vs_right"))
    print(i)
np.savetxt("real_acu.csv", scores, delimiter=",")
