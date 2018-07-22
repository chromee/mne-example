import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dt

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn import svm

import convert_mne_from_csv

def culc_by_csp_and_lda(path, csp_path, lda_path):
    epochs = convert_mne_from_csv.epochs_from_csv(path)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    with open(csp_path, 'rb') as pickle_file:
        csp = pickle.load(pickle_file)
    with open(lda_path, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)
    X_test = csp.transform(epochs_data)
    print(X_test.shape)
    score = lda.score(X_test, labels)
    print(path, score)

root = Path("./data/csv")
for path in root.iterdir():
    if path.is_file():
        cap_path = "./data/models/csp/csp_" + str(path.name) + ".pickle"
        lda_path = "./data/models/lda/lda_" + str(path.name) + ".pickle"
        culc_by_csp_and_lda(path, cap_path, lda_path)

# path = Path("./data/csv/jtanaka_MIK_14_05_2016_13_33_15_0000.csv")
# cap_path = "./data/models/csp_jtanaka_MIK_14_05_2016_13_33_15_0000.csv.pickle"
# lda_path = "./data/models/lda_jtanaka_MIK_14_05_2016_13_33_15_0000.csv.pickle"
# culc_by_csp_and_lda(path, cap_path, lda_path)