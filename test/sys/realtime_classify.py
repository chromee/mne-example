import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep

import mne
import convert_mne_from_csv

path = "./data/csv/jtanaka_MIK_14_05_2016_13_33_15_0000.csv"
data = np.loadtxt(path, delimiter=",", skiprows=1).T

sfreq = 512
interval = 1. / sfreq

with open("./data/models/csp/csp_jtanaka_MIK_14_05_2016_13_33_15_0000.csv_python36_20180717.pickle", 'rb') as pickle_file:
    csp = pickle.load(pickle_file)
with open("./data/models/lda/lda_jtanaka_MIK_14_05_2016_13_33_15_0000.csv_python36_20180717.pickle", 'rb') as pickle_file:
    lda = pickle.load(pickle_file)

w_length = int(sfreq * 0.5)
w_length = 2561
w_count = 9
w_step = int(sfreq * 0.1)

raws = []
step = 0

for i in range(len(data[0])):
    raws.append(data[1:17,i])
    step += 1
    if len(raws) > w_length * (w_count) and step > w_step:
        w = []
        for j in range(w_count):
            if j == 0:
                w.append(np.array(raws[-w_length*(j+1):]).T)
            else:
                w.append(np.array(raws[-w_length*(j+1):-w_length*j]).T)
        window = np.array(w)
        # print(window)
        X_test = csp.transform(window)
        label = lda.score(X_test, )
        print(i, label)
        step = 0

    # sleep(interval)

