import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

import mne
import convert_mne_from_csv

path = "./data/csv/misl/jtanaka_MIK_14_05_2016_13_33_15_0000.csv"
data = np.loadtxt(path, delimiter=",", skiprows=1).T

sfreq = 512
interval = 1. / sfreq

with open("./data/models/misl/csp/csp_jtanaka_MIK_14_05_2016_13_33_15_0000.csv.pickle", 'rb') as pickle_file:
    csp = pickle.load(pickle_file)
with open("./data/models/misl/svm/svm_jtanaka_MIK_14_05_2016_13_33_15_0000.csv.pickle", 'rb') as pickle_file:
    lda = pickle.load(pickle_file)

w_length = int(sfreq * 0.5)
# w_length = 2561
w_step = int(sfreq * 0.1)

step = 0
count = 0
score = 0

for i in range(len(data[0])):
    raw = data[1:17,:i+1]
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

