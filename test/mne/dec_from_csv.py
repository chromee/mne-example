import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import mne.decoding

def culc_by_csp_and_lda(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1).T
    ch_types = ["eeg" for i in range(16)] + ["stim"]
    ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "STIM"]
    # ch_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "STIM"]

    info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
    raw = mne.io.RawArray(data[1:], info)
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
    # raw.plot(n_channels=len(ch_names), scalings='auto', title='Data from arrays', show=True, block=True)

    events = mne.find_events(raw, stim_channel='STIM')
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    event_id = dict(right=1, left=2)
    epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
    # epochs.plot(picks=picks, scalings='auto', show=True, block=True)

    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 1

    # -----------------------LDAによる分類----------------------------

    # モンテカルロ相互検証ジェネレータを定義する（分散を減らす）：
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)   # crossvalidation(交差検証)の略で、データのsplitの方法を指定
    cv_split = cv.split(epochs_data_train)

    # 分類器を組み立てる
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    lda = svm.SVC()
    csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # cross_val_score関数（交差検証法の精度を出す関数）でscikit-learn Pipelineを使用する
    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('SVM', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # 交差検証法による分類精度の平均を表示
    class_balance = np.mean(labels == labels[0])    # 0と1のラベルの数の比率
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

    # # 視覚化のための完全なデータで推定されたCSPパターンのプロット
    # csp.fit_transform(epochs_data, labels)
    # layout = read_layout('EEG1005')
    # csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)

    # ---------------------------時間の経過とともにパフォーマンスを調べる------------------------------

    sfreq = raw.info['sfreq']   # サンプルレート
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)   # w_start = [ 0 51 102 153 ... 2295]
    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)    # データとラベルからフィルタを変形させる
        X_test = csp.transform(epochs_data_train[test_idx])                   # フィルタからデータを変形させる

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time :' + path.name)
    plt.legend(loc='lower right')
    plt.savefig(str(path) + "_svm.png")
    # plt.show()

root = Path("C:/Users/dk334/workspace/mne-example/data")
for path in root.iterdir():
    if path.is_file():
        culc_by_csp_and_lda(path)

# path = Path("C:/Users/dk334/workspace/mne-example/data/afujii_MIK_20_07_2017_17_00_48_0000.csv")
# culc_by_csp_and_lda(path)