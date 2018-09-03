import pickle
import numpy as np
from pathlib import Path
from datetime import datetime as dt

import sklearn.discriminant_analysis
from sklearn import svm

import mne
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.mne_wrapper import get_epochs

def save_model(subject, runs, event_id):
    epochs = get_epochs(subject, runs, event_id)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    _svm = svm.SVC()

    X_train = csp.fit_transform(epochs_data, labels)
    lda.fit(X_train, labels)
    _svm.fit(X_train, labels)

    file_name = "csp_subject" + str(subject) + ".pickle"
    file_path = "data/models/csp/" + file_name
    with open(file_path, mode='wb') as csp_file:
      pickle.dump(csp, csp_file)
    
    file_name = "lda_subject" + str(subject) + ".pickle"
    file_path = "data/models/lda/" + file_name
    with open(file_path, mode='wb') as lda_file:
        pickle.dump(lda, lda_file)

    file_name = "svm_subject" + str(subject) + ".pickle"
    file_path = "data/models/svm/" + file_name
    with open(file_path, mode='wb') as svm_file:
        pickle.dump(_svm, svm_file)

for i in range(1, 110):
    save_model(subject=i)

# save_model(subject=1)
print("ended")