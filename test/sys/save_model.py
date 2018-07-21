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
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    _svm = svm.SVC()

    X_train = csp.fit_transform(epochs_data, labels)
    lda.fit(X_train, labels)
    _svm.fit(X_train, labels)

    file_name = "csp_" + path.name + "_python36_" + dt.now().strftime('%Y%m%d') + ".pickle"
    file_path = "data/models/csp/" + file_name
    with open(file_path, mode='wb') as csp_file:
      pickle.dump(csp, csp_file)
    
    file_name = "lda_" + path.name + "_python36_" + dt.now().strftime('%Y%m%d') + ".pickle"
    file_path = "data/models/lda/" + file_name
    with open(file_path, mode='wb') as lda_file:
        pickle.dump(lda, lda_file)

    file_name = "svm_" + path.name + "_python36_" + dt.now().strftime('%Y%m%d') + ".pickle"
    file_path = "data/models/svm/" + file_name
    with open(file_path, mode='wb') as svm_file:
        pickle.dump(_svm, svm_file)

# root = Path("./data/csv")
# for path in root.iterdir():
#     if path.is_file():
#         save_model(path)

path = Path("./data/csv/jtanaka_MIK_14_05_2016_13_33_15_0000.csv")
save_model(path)