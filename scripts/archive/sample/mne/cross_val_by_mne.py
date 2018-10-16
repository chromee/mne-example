import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import KFold, ShuffleSplit, cross_val_score
from sklearn import svm

import mne
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

mne.set_log_level('WARNING')


def culc_by_csp_and_lda(subject, runs=[6, 10, 14], event_id=dict(hands=2, feet=3)):
    tmin, tmax = -1., 4.

    raw_fnames = eegbci.load_data(subject, runs)
    raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
                 raw_fnames]
    raw = concatenate_raws(raw_files)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events = find_events(raw, shortest_event=0, stim_channel='STI 014')

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    # -----------------------LDAによる分類----------------------------

    # モンテカルロ相互検証ジェネレータを定義する（分散を減らす）：
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()

    cv = KFold(n_splits=5)
    # cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    # lda = svm.SVC()
    csp = mne.decoding.CSP(n_components=4, reg=None,
                           log=True, norm_trace=False)

    clf = sklearn.pipeline.Pipeline([('CSP', csp), ('SVM', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    class_balance = np.mean(labels == labels[0])    # 0と1のラベルの数の比率
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" %
          (np.mean(scores), class_balance))

    # # 視覚化のための完全なデータで推定されたCSPパターンのプロット
    csp.fit_transform(epochs_data, labels)
    layout = read_layout('EEG1005')
    csp.plot_patterns(epochs.info, layout=layout,
                      ch_type='eeg', units='Patterns (AU)', size=1.5)

    # ---------------------------時間の経過とともにパフォーマンスを調べる------------------------------

    sfreq = raw.info['sfreq']   # サンプルレート
    w_length = int(sfreq * 0.5)   # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
    scores_windows = []

    for train_idx, test_idx in cv.split(epochs_data_train):
        y_train, y_test = labels[train_idx], labels[test_idx]

        # ----------- train---------------
        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        lda.fit(X_train, y_train)

        # ------------test---------------
        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            data = epochs_data[test_idx][:, :, n:(n + w_length)]
            X_test = csp.transform(data)
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
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    # plt.savefig(str(path) + "_svm.png")
    plt.show()


culc_by_csp_and_lda(1)

# for i in range(1, 110):
#     culc_by_csp_and_lda(i)
