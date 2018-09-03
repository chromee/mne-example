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
with open("./data/models/csp/csp_subject" + str(subject) + ".pickle", 'rb') as pickle_file:
    lda = pickle.load(pickle_file)

w_length = int(sfreq * 0.5)
# w_length = 2561
w_step = int(sfreq * 0.1)

step = 0
count = 0
score = 0

for i in range(len(data[0])):
    raw = data[1:17, :i+1]
    step += 1
    if i > w_length and step > w_step:
        window = np.array([raw[:, -w_length:]])
        X_test = csp.transform(window)
        label = lda.predict(X_test)
        step = 0
        print(i, label)
        # if label == data[17][i]: score += 1
        # count += 1
    # sleep(interval)
# print(score/count)
