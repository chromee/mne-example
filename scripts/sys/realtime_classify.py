import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep
import mne
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_raw

subject = 1
sfreq = 512
interval = 1. / sfreq

raw = get_raw(subject)
data = raw.get_data()

with open("./data/models/csp/csp_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
    csp = pickle.load(pickle_file)
with open("./data/models/lda/lda_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
    lda = pickle.load(pickle_file)
with open("./data/models/svm/svm_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
    svm = pickle.load(pickle_file)

w_length = 161  # 学習epochのlength
w_step = int(sfreq)

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

        print(i, label, argmax_label)

        if label == argmax_label:
            score += 1
    sleep(interval)
print(score/count)
