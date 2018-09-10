import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime as dt

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn import svm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mylib.mne_wrapper import get_epochs


def culc_by_csp_and_lda(subject):
    epochs = get_epochs(subject)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    csp_path = "./data/models/csp/csp_subject" + str(subject) + ".pickle"
    with open(csp_path, 'rb') as pickle_file:
        csp = pickle.load(pickle_file)
    lda_path = "./data/models/lda/lda_subject" + str(subject) + ".pickle"
    with open(lda_path, 'rb') as pickle_file:
        lda = pickle.load(pickle_file)
    X_test = csp.transform(epochs_data)
    score = lda.score(X_test, labels)
    print("subject"+str(subject), score)


culc_by_csp_and_lda(subject=1)

# for i in range(1, 110):
#     culc_by_csp_and_lda(subject=i)
