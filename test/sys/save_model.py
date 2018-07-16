import pickle
import numpy as np
from pathlib import Path
from datetime import datetime as dt

import sklearn.discriminant_analysis
from sklearn import svm

import mne
import convert_mne_from_csv

def save_model(path):
    epochs = convert_mne_from_csv.epochs_from_csv(path)
    epochs_data = epochs.get_data()
    labels = epochs.events[:, -1] - 1

    # 分類器を組み立てる
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)

    X_train = csp.fit_transform(epochs_data, labels)
    lda.fit(X_train, labels)

    tail = "_eeg_python36_" + path.name + "_" + dt.now().strftime('%Y%m%d') + ".pickle"
    file_name = "data/models/csp" + tail
    with open(file_name, mode='wb') as csp_file:
      pickle.dump(csp, csp_file)

    file_name = "data/models/lda" + tail
    with open(file_name, mode='wb') as lda_file:
        pickle.dump(lda, lda_file)

# root = Path("./data/csv")
# for path in root.iterdir():
#     if path.is_file():
#         save_model(path)

path = Path("./data/csv/afujii_MIK_20_07_2017_17_00_48_0000.csv")
save_model(path)