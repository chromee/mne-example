# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import sklearn.svm
import sklearn.pipeline
import sklearn.discriminant_analysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import mne.decoding

# print(__doc__)

mne.set_log_level('WARNING')

# #############################################################################
# # Set parameters and read data

# 開始後1秒後に始まるエポックを用いて誘発反応の分類を避ける
event_id = dict(hands=2, feet=3)
subject = 1         # subjectは1~109ある
runs = [6, 10, 14]  # motor imagery: hands vs feet (https://www.martinos.org/mne/stable/generated/mne.datasets.eegbci.load_data.html?highlight=load_data)

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))     # ['Fc5.', 'Fc3.', ... ] -> ['Fc5', 'Fc3', ... ]

# Apply band-pass filter (raw.infoにて，lowpass 80.0 -> 30.0Hz ,  highpass 0.0 -> 7.0Hz)
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')     # stim:自己刺激？
# 各イベント(1~3)がどの時間に行うかプロット
# mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)

# picks=[0, 1, 2, ... , 63] 全チャンネルの配列の中からEEGのチャンネルのindexを取得
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# エポックを読み込む（訓練は1〜2秒間のみ実行される）テストは実行中の分類器で行われる
# tmin, tmax : イベントの開始・終了時間．それぞれデフォは -0.2, 0.5 
# proj=True  : SSP投影ベクトルを適用する（？）
epochs = Epochs(raw, events, event_id, tmin=-1., tmax=4., proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1] - 2   # イベントのラベルが[2, 3]なのを[0, 1]に変換

# # -----------------------LDAによる分類----------------------------

# モンテカルロ相互検証ジェネレータを定義する（分散を減らす）：
scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# # 分類器を組み立てる
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
csp = mne.decoding.CSP(n_components=4, reg=None, log=True, norm_trace=False)

# cross_val_score関数(交差検証をする関する)でscikit-learn Pipelineを使用する
clf = sklearn.pipeline.Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# 分類精度の表示
class_balance = np.mean(labels == labels[0])    # 0と1の割合（ラベルの存在比率）
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

# 視覚化のための完全なデータで推定されたCSPパターンのプロット
csp.fit_transform(epochs_data, labels)

layout = read_layout('EEG1005')
# csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)
# csp.plot_filters(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)
print(csp.filters_+csp.filters_)     # filterの取得

# # ---------------------------時間の経過とともにパフォーマンスを調べる------------------------------

# sfreq = raw.info['sfreq']
# w_length = int(sfreq * 0.5)   # running classifier: window length
# w_step = int(sfreq * 0.1)  # running classifier: window step size
# w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

# scores_windows = []

# for train_idx, test_idx in cv_split:
#     y_train, y_test = labels[train_idx], labels[test_idx]

#     X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
#     X_test = csp.transform(epochs_data_train[test_idx])

#     # fit classifier
#     lda.fit(X_train, y_train)

#     # running classifier: test classifier on sliding window
#     score_this_window = []
#     for n in w_start:
#         X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
#         score_this_window.append(lda.score(X_test, y_test))
#     scores_windows.append(score_this_window)

# # Plot scores over time
# w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

# plt.figure()
# plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
# plt.axvline(0, linestyle='--', color='k', label='Onset')
# plt.axhline(0.5, linestyle='-', color='k', label='Chance')
# plt.xlabel('time (s)')
# plt.ylabel('classification accuracy')
# plt.title('Classification score over time')
# plt.legend(loc='lower right')
# plt.show()